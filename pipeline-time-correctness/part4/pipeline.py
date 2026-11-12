"""A small daily pipeline for late-arriving counts, in DuckDB.

Three layers:
  landing  append-only. Every file received is recorded once per content hash, with the time it
           was loaded; its rows are copied unchanged. Nothing here is ever updated or deleted.
  core     views and tables derived from landing as of a point in time: what the pipeline knew
           (known_at), and the delay profile it estimated from that.
  mart     one row per run: the 7-day count as published, a nowcast, an interval from the
           pipeline's own past errors, the estimated completeness, week-over-week changes at
           equal maturity, and the time and code version that computed it.

Every derived value is a function of (landing rows loaded by a time, code version). A run for a
date therefore replaces its own partition only, and any past run can be recomputed exactly.
"""
from __future__ import annotations

import datetime as dt
import hashlib
from dataclasses import dataclass
from pathlib import Path

import duckdb
import pandas as pd

SCHEMA_SQL = """
create schema if not exists landing;
create schema if not exists core;
create schema if not exists mart;

create table if not exists landing.files (
    file_name        varchar   not null,
    sha256           varchar   not null,
    publication_date date      not null,   -- the source's date for the file (its "Datenstand")
    n_rows           integer   not null,
    loaded_at        timestamp not null,   -- when this pipeline received it
    primary key (file_name, sha256)
);

create table if not exists landing.cases (
    file_name        varchar   not null,
    sha256           varchar   not null,
    publication_date date      not null,
    state            integer   not null,
    reporting_date   date      not null,
    cases            integer   not null,
    loaded_at        timestamp not null
);

create table if not exists core.delay_profile (
    computed_at      timestamp not null,
    weekday          integer   not null,   -- ISO weekday of the reporting date, 1 = Monday
    age              integer   not null,   -- days from the reporting date to the publication
    share            double,               -- share of the 28-day count known at this age
    mean_full        double,               -- mean 28-day count of the dates used
    n_dates          integer   not null
);

create table if not exists mart.daily_report (
    run_date           date      not null primary key,
    as_of_publication  date      not null,
    stale              boolean   not null,
    published_7d       bigint    not null,
    nowcast_7d         double    not null,
    lower_7d           double,
    upper_7d           double,
    completeness       double    not null,
    wow_published      double,
    wow_nowcast        double,
    computed_at        timestamp not null,
    code_version       varchar   not null
);

-- what the pipeline had loaded by `as_of`: for each publication, the latest version loaded by then
create or replace macro core.known_at(as_of) as table
with f as (
    select file_name, sha256,
           row_number() over (partition by publication_date order by loaded_at desc, sha256) as rn
    from landing.files
    where loaded_at <= as_of
)
select c.publication_date, c.reporting_date, sum(c.cases) as cases
from landing.cases c
join f on f.file_name = c.file_name and f.sha256 = c.sha256 and f.rn = 1
group by c.publication_date, c.reporting_date;
"""

STATE_MACRO_SQL = """
create or replace macro core.known_by_state_at(as_of) as table
with f as (
    select file_name, sha256,
           row_number() over (partition by publication_date order by loaded_at desc, sha256) as rn
    from landing.files
    where loaded_at <= as_of
)
select c.publication_date, c.state, c.reporting_date, sum(c.cases) as cases
from landing.cases c
join f on f.file_name = c.file_name and f.sha256 = c.sha256 and f.rn = 1
group by c.publication_date, c.state, c.reporting_date;
"""

# Training rows for a model that predicts next week's cases in a state from this week's, built at
# $build_at. The feature is taken twice: as published on the run day, and as it stood 28 days
# after each date. The label is next week's count 28 days after each date.
TRAINING_ROWS_SQL = """
with k as (select * from core.known_by_state_at($build_at)),
last_pub as (select max(publication_date) as p from k),
runs as (
    select distinct publication_date as r from k, last_pub
    where publication_date between $first and last_pub.p - 35
),
states as (select distinct state from k),
cells as (
    select runs.r, states.state, runs.r - i.i::integer as d, 'feature' as role
    from runs, states, (select unnest(range(1, 8)) as i) i
    union all
    select runs.r, states.state, runs.r + i.i::integer as d, 'label' as role
    from runs, states, (select unnest(range(0, 7)) as i) i
),
as_published as (                                -- what the run day's file said
    select c.r, c.state, sum(coalesce(k.cases, 0)) as x_as_published
    from cells c
    left join k on k.publication_date = c.r and k.state = c.state and k.reporting_date = c.d
    where c.role = 'feature'
    group by c.r, c.state
),
at_28 as (                                       -- the same dates as they stood 28 days later
    select c.r, c.state, c.role, coalesce(k.cases, 0) as cases
    from (select *, d + 28 as on_day from cells) c
    asof left join k on k.state = c.state and k.reporting_date = c.d and c.on_day >= k.publication_date
)
select a.r as run_date, a.state, p.x_as_published,
       sum(case when a.role = 'feature' then a.cases end) as x_final,
       sum(case when a.role = 'label' then a.cases end) as y
from at_28 a join as_published p on p.r = a.r and p.state = a.state
group by a.r, a.state, p.x_as_published
order by a.r, a.state
"""

# The delay profile as of a time: for each weekday, the share of the 28-day count known at each
# age, from the four latest reporting dates of that weekday whose 28-day count is known.
PROFILE_SQL = """
with k as (select * from core.known_at($as_of)),
latest as (select max(publication_date) as p from k),
complete as (
    select distinct k.reporting_date as d
    from k, latest
    where k.reporting_date between latest.p - 55 and latest.p - 28
),
pick as (
    select d, isodow(d) as weekday
    from (select d, row_number() over (partition by isodow(d) order by d desc) as rk from complete)
    where rk <= 4
),
grid as (
    select pick.d, pick.weekday, a.age, pick.d + a.age::integer as on_day
    from pick cross join (select unnest(range(0, 29)) as age) a
),
vals as (                                            -- each date's count as it stood at each age
    select g.d, g.weekday, g.age, coalesce(k.cases, 0) as cases
    from grid g
    asof left join k on k.reporting_date = g.d and g.on_day >= k.publication_date
),
full28 as (select d, cases as full from vals where age = 28)
select $as_of::timestamp as computed_at, v.weekday, v.age,
       sum(v.cases) / nullif(sum(f.full), 0) as share,
       avg(f.full) as mean_full,
       count(*)::integer as n_dates
from vals v join full28 f using (d)
group by v.weekday, v.age
order by v.weekday, v.age
"""

# The report for one run. Uses only landing rows loaded by $as_of and mart rows of earlier runs.
REPORT_SQL = """
with k as (select * from core.known_at($as_of)),
latest as (select max(publication_date) as p from k),
profile as (select * from core.delay_profile where computed_at = $as_of::timestamp),
win as (                                             -- the seven reporting dates before the publication
    select latest.p as pub, (latest.p - w.i::integer) as d, w.i::integer as age
    from latest cross join (select unnest(range(1, 8)) as i) w
),
cur as (
    select win.*, isodow(win.d) as weekday, coalesce(k.cases, 0) as known
    from win left join k on k.publication_date = win.pub and k.reporting_date = win.d
),
est as (                                             -- known / expected share; a level when too little is in
    select cur.*,
           case when pr.share >= 0.2 then cur.known / pr.share else pr.mean_full end as nowcast
    from cur join profile pr on pr.weekday = cur.weekday and pr.age = cur.age
),
past_runs as (                                       -- earlier runs whose seven dates are now 28 days old
    select r.run_date, r.nowcast_7d, r.as_of_publication
    from mart.daily_report r, latest
    where r.computed_at < $as_of::timestamp
      and r.as_of_publication <= latest.p - 29
    order by r.run_date desc
    limit 56
),
past_dates as (
    select run_date, d, d + 28 as at_28
    from (select run_date, as_of_publication - unnest(range(1, 8))::integer as d from past_runs)
),
past_values as (
    select pd.run_date, coalesce(v.cases, 0) as cases
    from past_dates pd
    asof left join k v on v.reporting_date = pd.d and pd.at_28 >= v.publication_date
),
past as (
    select pr.run_date, pr.nowcast_7d, m.matured
    from past_runs pr
    join (select run_date, sum(cases) as matured from past_values group by run_date) m using (run_date)
),
band as (
    select quantile_cont(matured / nowcast_7d, 0.05) as q05,
           quantile_cont(matured / nowcast_7d, 0.95) as q95,
           count(*) as n
    from past
),
prev as (select * from mart.daily_report
         where run_date = $run_date::date - 7 and computed_at < $as_of::timestamp)
select $run_date::date as run_date,
       max(est.pub) as as_of_publication,
       max(est.pub) < $run_date::date as stale,
       sum(est.known)::bigint as published_7d,
       sum(est.nowcast) as nowcast_7d,
       case when max(band.n) >= 28 then sum(est.nowcast) * max(band.q05) end as lower_7d,
       case when max(band.n) >= 28 then sum(est.nowcast) * max(band.q95) end as upper_7d,
       sum(est.known) / sum(est.nowcast) as completeness,
       sum(est.known) / max(prev.published_7d) - 1 as wow_published,
       sum(est.nowcast) / max(prev.nowcast_7d) - 1 as wow_nowcast,
       $as_of::timestamp as computed_at,
       $code_version as code_version
from est cross join band left join prev on true
"""

CODE_VERSION = hashlib.sha256((SCHEMA_SQL + PROFILE_SQL + REPORT_SQL).encode()).hexdigest()[:12]
RUN_TIME = dt.time(8, 0)  # the daily run, after the source publishes in the morning


# ---------------------------------------------------------------------------
# Two optional changes, added after Part 4 (the follow-up). With both off, the SQL is exactly the
# Part 4 SQL above and the code version is the same.
#   content_check  find the latest file that brought anything new (the "data cut") and count the
#                  age of each reporting date from that file, not from the latest publication. Dates
#                  after the cut are then treated as unknown (the level is used), not as zero. The
#                  report gains a column, data_cut.
#   holidays       treat Germany's nationwide public holidays as Sundays, in the delay profile and in
#                  the report.
#   graded         (Part 6) a number tau, or None. Instead of asking whether a file brought anything,
#                  ask how much of what it was expected to bring it brought: for each of the 14 latest
#                  files, the cases it added against the cases the delay profile expected it to add
#                  (the increase in each date's expected share, times an estimate of the date's final
#                  count from a week or more back). A file that brought less than tau of that is short:
#                  the share it was expected to add is taken off the dates it covers, in proportion to
#                  what is missing, unless later files brought the missing cases. Needs
#                  fallback="last_week" and no content_check. The report gains a column,
#                  missing_cases.
#   fallback       what to use for a date with less than a fifth of its count expected in hand:
#                  "level" (Part 4), the mean 28-day count of the four dates in the profile, 28 to 55
#                  days old; or "last_week", the estimate for the latest earlier date of the same
#                  weekday that is at least seven days old.
# ---------------------------------------------------------------------------
def easter(year: int) -> dt.date:
    """Gregorian Easter Sunday (anonymous Gregorian algorithm)."""
    a, b, c = year % 19, year // 100, year % 100
    d, e = b // 4, b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = c // 4, c % 4
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    month = (h + l - 7 * m + 114) // 31
    day = (h + l - 7 * m + 114) % 31 + 1
    return dt.date(year, month, day)


def german_holidays(first_year: int = 2020, last_year: int = 2024) -> list[dt.date]:
    """The nine public holidays observed in every German state."""
    out = []
    for y in range(first_year, last_year + 1):
        e = easter(y)
        out += [dt.date(y, 1, 1), e - dt.timedelta(days=2), e + dt.timedelta(days=1), dt.date(y, 5, 1),
                e + dt.timedelta(days=39), e + dt.timedelta(days=50), dt.date(y, 10, 3),
                dt.date(y, 12, 25), dt.date(y, 12, 26)]
    return sorted(out)


def holiday_sql(years: tuple[int, int] = (2020, 2024)) -> str:
    values = ", ".join(f"(date '{d.isoformat()}')" for d in german_holidays(*years))
    return f"""
create table if not exists core.holidays (d date primary key);
insert or ignore into core.holidays values {values};
-- the weekday used by the profile and the report: a holiday counts as a Sunday (7)
create or replace macro core.daytype(x) as case when x in (select d from core.holidays) then 7 else isodow(x) end;
"""


def _replace_once(text: str, old: str, new: str) -> str:
    assert text.count(old) == 1, old
    return text.replace(old, new)


def variant_sql(content_check: bool = False, holidays: bool = False, fallback: str = "level",
                graded: float | None = None, holiday_years: tuple[int, int] = (2020, 2024)
                ) -> tuple[str, str, str, str]:
    """(schema, extra, profile, report) SQL for a variant. The defaults give the Part 4 SQL."""
    assert fallback in ("level", "last_week")
    assert graded is None or (fallback == "last_week" and not content_check), "graded needs last_week, no content_check"
    schema, extra, profile, report = SCHEMA_SQL, "", PROFILE_SQL, REPORT_SQL
    if holidays:
        extra = holiday_sql(holiday_years)   # Part 5 used 2020-2024; Part 6 extends it to 2027
        profile = profile.replace("isodow(d)", "core.daytype(d)")
        report = _replace_once(report, "isodow(win.d)", "core.daytype(win.d)")
    if content_check:
        schema = _replace_once(schema, "    code_version       varchar   not null\n);",
                               "    code_version       varchar   not null,\n    data_cut           date\n);")
        report = _replace_once(report, "win as (", """pubs as (select publication_date as q, lag(publication_date) over (order by publication_date) as prev
         from (select distinct publication_date from k)),
changes as (                                         -- what each file changed against the file before it
    select a.publication_date as q, sum(abs(a.cases - coalesce(b.cases, 0))) as change
    from k a
    join pubs on pubs.q = a.publication_date and pubs.prev is not null
    left join k b on b.publication_date = pubs.prev and b.reporting_date = a.reporting_date
    group by a.publication_date
),
cut as (                                             -- the latest file that brought anything new
    select coalesce(max(changes.q), max(latest.p)) as e
    from latest left join changes on changes.q <= latest.p and changes.change > 0
),
win as (""")
        report = _replace_once(report, """coalesce(k.cases, 0) as known
    from win left join k""", """coalesce(k.cases, 0) as known,
           greatest(cut.e - win.d, 0)::integer as age_eff  -- age counted from the data cut
    from win cross join cut left join k""")
        report = _replace_once(report, "pr.weekday = cur.weekday and pr.age = cur.age",
                               "pr.weekday = cur.weekday and pr.age = cur.age_eff")
        report = _replace_once(report, """       $code_version as code_version
from est cross join band left join prev on true""", """       $code_version as code_version,
       max(cut.e) as data_cut
from est cross join band cross join cut left join prev on true""")
    if fallback == "last_week":
        daytype = "core.daytype" if holidays else "isodow"
        ref, cutjoin = ("cut.e", "cross join cut") if content_check else ("latest.p", "")
        report = _replace_once(report, "\nest as (", f"""\nhist as (                                            -- the 35 dates before, each with its share
    select h.d, h.weekday, h.age, h.known, pr.share
    from (
        select latest.p - i.i::integer as d, {daytype}(latest.p - i.i::integer) as weekday,
               ({ref} - (latest.p - i.i::integer))::integer as age, coalesce(k.cases, 0) as known
        from latest {cutjoin} cross join (select unnest(range(1, 36)) as i) i
        left join k on k.publication_date = latest.p and k.reporting_date = latest.p - i.i::integer
    ) h
    join profile pr on pr.weekday = h.weekday and pr.age = least(greatest(h.age, 0), 28)
),
est as (""")
        report = _replace_once(report, "case when pr.share >= 0.2 then cur.known / pr.share else pr.mean_full end as nowcast",
                               """case when pr.share >= 0.2 then cur.known / pr.share
                else coalesce((select hs.known / hs.share from hist hs       -- same weekday, a week or more old
                               where hs.weekday = cur.weekday and hs.d < cur.d and hs.age >= 7 and hs.share >= 0.2
                               order by hs.d desc limit 1), pr.mean_full) end as nowcast""")
    if graded is not None:
        daytype = "core.daytype" if holidays else "isodow"
        tau = repr(float(graded))
        schema = _replace_once(schema, "    code_version       varchar   not null\n);",
                               "    code_version       varchar   not null,\n    missing_cases      double\n);")
        report = _replace_once(report, "win as (", f"""gfiles as (                                          -- the 14 latest files, each with the file before it
    select q, prev
    from (select publication_date as q, lag(publication_date) over (order by publication_date) as prev
          from (select distinct publication_date from k)) f, latest
    where f.q > latest.p - 14 and f.prev is not null
),
gdates as (                                          -- what each of them added to each of the 28 dates it covers
    select f.q, f.prev, f.q - i.i::integer as d,
           coalesce(k1.cases, 0) - coalesce(k0.cases, 0) as added
    from gfiles f cross join (select unnest(range(1, 29)) as i) i
    left join k k1 on k1.publication_date = f.q and k1.reporting_date = f.q - i.i::integer
    left join k k0 on k0.publication_date = f.prev and k0.reporting_date = f.q - i.i::integer
),
glevel_base as (                                     -- the 49 dates before the latest file
    select latest.p - i.i::integer as d, {daytype}(latest.p - i.i::integer) as weekday, i.i::integer as age,
           coalesce(k.cases, 0) as known
    from latest cross join (select unnest(range(1, 50)) as i) i
    left join k on k.publication_date = latest.p and k.reporting_date = latest.p - i.i::integer
),
glevel_known as (                                    -- dates a week or more old: known / expected share
    select b.d, b.weekday, b.known / pr.share as level
    from glevel_base b join profile pr on pr.weekday = b.weekday and pr.age = least(b.age, 28)
    where b.age >= 7 and pr.share >= 0.2
),
glevel as (                                          -- younger dates: the latest such date of the same type
    select b.d, coalesce((select lk.level from glevel_known lk where lk.d = b.d),
                         (select lk.level from glevel_known lk where lk.weekday = b.weekday and lk.d < b.d
                          order by lk.d desc limit 1), 0) as level
    from glevel_base b
),
gflow as (                                           -- the share of each date each file was expected to add
    select g.q, g.d, g.added, lv.level,
           greatest(coalesce(p1.share, 0) - coalesce(p0.share, 0), 0) as dshare
    from gdates g
    join glevel lv on lv.d = g.d
    left join profile p1 on p1.weekday = {daytype}(g.d) and p1.age = least(greatest(g.q - g.d, 0), 28)
    left join profile p0 on p0.weekday = {daytype}(g.d) and p0.age = least(greatest(g.prev - g.d, 0), 28)
),
gfile as (                                           -- per file: added against expected
    select q, sum(added) as observed, sum(dshare * level) as expected,
           case when sum(dshare * level) > 0 then sum(added) / sum(dshare * level) else 1 end as r
    from gflow group by q
),
gw as (select q, expected, r, case when r >= {tau} then 1.0 else greatest(r, 0.0) end as w from gfile),
gseq as (                                            -- running balance: cases missing minus cases brought beyond expectation
    select q, (1 - w) * expected as short,
           sum((1 - w) * expected - greatest(r - 1, 0) * expected) over (order by q) as cum
    from gw
),
gback as (                                           -- the backlog now; it restarts wherever the balance was lowest
    select (select cum from gseq order by q desc limit 1) - least(0, min(cum)) as backlog, least(0, min(cum)) as m
    from gseq
),
greset as (
    select coalesce(max(gseq.q) filter (where gseq.cum <= gback.m + 1e-9), date '1900-01-01') as q0,
           max(gback.backlog) as backlog
    from gseq, gback
),
gpi as (                                             -- the part of the shortfall since then not yet made up
    select case when sh.short > 0 then least(1.0, gr.backlog / sh.short) else 0.0 end as f, gr.backlog
    from greset gr,
         (select coalesce(sum(gseq.short), 0) as short from gseq, greset where gseq.q > greset.q0) sh
),
gmiss as (
    select f.d, sum((1 - gw.w) * f.dshare) as missing_share
    from gflow f join gw using (q), greset
    where f.q > greset.q0
    group by f.d
),
win as (""")
        report = _replace_once(report, """coalesce(k.cases, 0) as known
    from win left join k""", """coalesce(k.cases, 0) as known,
           gpi.f * coalesce(gm.missing_share, 0) as missing_share,
           gpi.backlog as missing_cases
    from win cross join gpi left join gmiss gm on gm.d = win.d left join k""")
        report = _replace_once(report, "case when pr.share >= 0.2 then cur.known / pr.share",
                               "case when pr.share - cur.missing_share >= 0.2 then cur.known / (pr.share - cur.missing_share)")
        report = _replace_once(report, """       $code_version as code_version
from est""", """       $code_version as code_version,
       max(est.missing_cases) as missing_cases
from est""")
    return schema, extra, profile, report


def run_ts(day: dt.date) -> dt.datetime:
    return dt.datetime.combine(day, RUN_TIME)


@dataclass(frozen=True)
class Incoming:
    path: Path
    publication_date: dt.date
    arrives_at: dt.datetime


class Pipeline:
    def __init__(self, database: str = ":memory:", content_check: bool = False, holidays: bool = False,
                 fallback: str = "level", graded: float | None = None,
                 holiday_years: tuple[int, int] = (2020, 2024)):
        self.con = duckdb.connect(database)
        self.con.execute("set threads = 1")
        schema, extra, self.profile_sql, self.report_sql = variant_sql(content_check, holidays, fallback, graded,
                                                                       holiday_years)
        self.code_version = hashlib.sha256((schema + extra + self.profile_sql + self.report_sql).encode()).hexdigest()[:12]
        self.con.execute(schema)
        self.con.execute(STATE_MACRO_SQL)
        if extra:
            self.con.execute(extra)

    # -- landing ---------------------------------------------------------------------------
    def land(self, item: Incoming, loaded_at: dt.datetime) -> bool:
        """Record a file and copy its rows. A file already landed with the same content is skipped."""
        data = item.path.read_bytes()
        sha = hashlib.sha256(data).hexdigest()
        exists = self.con.execute("select count(*) from landing.files where file_name = ? and sha256 = ?",
                                  [item.path.name, sha]).fetchone()[0]
        if exists:
            return False
        self.con.execute("begin transaction")
        self.con.execute("""
            insert into landing.cases
            select $name, $sha, $pub, state, reporting_date, cases, $at
            from read_csv($path, header = true,
                          columns = {'state': 'INTEGER', 'reporting_date': 'DATE', 'cases': 'INTEGER'})
        """, {"name": item.path.name, "sha": sha, "pub": item.publication_date, "at": loaded_at,
              "path": str(item.path)})
        n = self.con.execute("select count(*) from landing.cases where file_name = ? and sha256 = ?",
                             [item.path.name, sha]).fetchone()[0]
        self.con.execute("insert into landing.files values (?, ?, ?, ?, ?)",
                         [item.path.name, sha, item.publication_date, n, loaded_at])
        self.con.execute("commit")
        return True

    def load_incoming(self, incoming: list[Incoming], now: dt.datetime) -> list[str]:
        """Incremental load on arrival time.

        Reads every file that has arrived by `now` and either has a name not landed yet or arrived
        after the previous load (a file republished under the same name). `land` then skips any file
        whose content is already landed.
        """
        names = {r[0] for r in self.con.execute("select distinct file_name from landing.files").fetchall()}
        last = self.con.execute("select max(loaded_at) from landing.files").fetchone()[0]
        landed = []
        for item in sorted(incoming, key=lambda i: (i.arrives_at, i.publication_date)):
            if item.arrives_at > now:
                continue
            if item.path.name in names and last is not None and item.arrives_at <= last:
                continue
            if self.land(item, now):
                landed.append(item.path.name)
        return landed

    # -- core and mart ---------------------------------------------------------------------
    def build_profile(self, as_of: dt.datetime) -> None:
        self.con.execute("begin transaction")
        self.con.execute("delete from core.delay_profile where computed_at = ?", [as_of])
        self.con.execute("insert into core.delay_profile " + self.profile_sql, {"as_of": as_of})
        self.con.execute("commit")

    def report_row(self, day: dt.date, as_of: dt.datetime) -> pd.DataFrame:
        return self.con.execute(self.report_sql, {"as_of": as_of, "run_date": day,
                                                  "code_version": self.code_version}).df()

    def build_report(self, day: dt.date, as_of: dt.datetime) -> None:
        row = self.report_row(day, as_of)
        self.con.execute("begin transaction")
        self.con.execute("delete from mart.daily_report where run_date = ?", [day])
        self.con.register("new_row", row)
        self.con.execute("insert into mart.daily_report select * from new_row")
        self.con.unregister("new_row")
        self.con.execute("commit")

    def run(self, day: dt.date, incoming: list[Incoming]) -> list[str]:
        now = run_ts(day)
        landed = self.load_incoming(incoming, now)
        self.build_profile(now)
        self.build_report(day, now)
        return landed

    def reproduce(self, day: dt.date) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Recompute a past run from landing and earlier mart rows, without storing anything.

        Returns (profile, report row) as they come out now, to compare with the stored ones.
        """
        as_of = self.con.execute("select computed_at from mart.daily_report where run_date = ?", [day]).fetchone()[0]
        profile = self.con.execute(self.profile_sql, {"as_of": as_of}).df()
        return profile, self.report_row(day, as_of)

    def stored(self, day: dt.date) -> tuple[pd.DataFrame, pd.DataFrame]:
        as_of = self.con.execute("select computed_at from mart.daily_report where run_date = ?", [day]).fetchone()[0]
        profile = self.con.execute("select * from core.delay_profile where computed_at = ? order by weekday, age",
                                   [as_of]).df()
        row = self.con.execute("select * from mart.daily_report where run_date = ?", [day]).df()
        return profile, row

    def training_rows(self, build_at: dt.datetime, first: dt.date) -> pd.DataFrame:
        return self.con.execute(TRAINING_ROWS_SQL, {"build_at": build_at, "first": first}).df()

    def report(self) -> pd.DataFrame:
        return self.con.execute("select * from mart.daily_report order by run_date").df()


def same(a: pd.DataFrame, b: pd.DataFrame) -> bool:
    """Exact equality of values (dtypes aside): every cell's text form, including full float repr."""
    if list(a.columns) != list(b.columns) or a.shape != b.shape:
        return False
    def text(df: pd.DataFrame):
        return df.reset_index(drop=True).map(lambda v: "NA" if pd.isna(v) else str(v)).to_numpy()
    return bool((text(a) == text(b)).all())


# ---------------------------------------------------------------------------
# The source: the RKI files of Part 3, one file per publication day
# ---------------------------------------------------------------------------
PART3_DATA = Path(__file__).resolve().parents[1] / "part3" / "data"
PART6_DATA = Path(__file__).resolve().parents[1] / "part6" / "data"
DAILY_LAST_PART3 = dt.date(2023, 12, 31)   # Part 3 read daily files to here, then one a month


def write_incoming(directory: Path, first: dt.date, last: dt.date, delays: dict[dt.date, int] | None = None
                   ) -> list[Incoming]:
    """Write one CSV per RKI publication (cases by state and reporting date, the 28 days before).

    The source publishes each file in the morning of its date; `delays` holds files that reach the
    pipeline that many days late.
    """
    delays = delays or {}
    directory.mkdir(parents=True, exist_ok=True)
    st = pd.read_csv(PART3_DATA / "rki_state.csv.gz", parse_dates=["publication", "date"])
    if last > DAILY_LAST_PART3:            # Part 6: the daily files of 2024 to 2026, fetched separately
        st = pd.concat([st[st.publication <= pd.Timestamp(DAILY_LAST_PART3)],
                        pd.read_csv(PART6_DATA / "rki_state_2024_2026.csv.gz", parse_dates=["publication", "date"])])
    st = st[(st.publication >= pd.Timestamp(first)) & (st.publication <= pd.Timestamp(last))]
    out = []
    for pub, g in st.groupby("publication"):
        day = pub.date()
        path = directory / f"rki_{day.isoformat()}.csv"
        g[["state", "date", "cases"]].rename(columns={"date": "reporting_date"}).sort_values(
            ["state", "reporting_date"]).to_csv(path, index=False)
        arrives = dt.datetime.combine(day + dt.timedelta(days=delays.get(day, 0)), dt.time(7, 0))
        out.append(Incoming(path, day, arrives))
    return out


def replay(first_run: dt.date, last_run: dt.date, incoming: list[Incoming], database: str = ":memory:",
           content_check: bool = False, holidays: bool = False, fallback: str = "level",
           graded: float | None = None, holiday_years: tuple[int, int] = (2020, 2024)) -> Pipeline:
    p = Pipeline(database, content_check=content_check, holidays=holidays, fallback=fallback, graded=graded,
                 holiday_years=holiday_years)
    day = first_run
    while day <= last_run:
        p.run(day, incoming)
        day += dt.timedelta(days=1)
    return p


def final_national() -> pd.Series:
    """Cases by reporting date in the RKI file of 10 November 2026 (Part 3), for evaluation only."""
    s = pd.read_csv(PART3_DATA / "rki_snapshots.csv.gz", parse_dates=["publication", "date"])
    last = s.publication.max()
    return s[s.publication == last].set_index("date")["cases"]
