# Stage 2: package-native dependency spine

The canonical gap-aware and day-influence implementations now depend directly on package modules rather than historical project-root compatibility shims.

The dependency spine is:

```text
blood_pressure_missingness.public_analysis
    -> blood_pressure_missingness.analyses.gap_aware
        -> blood_pressure_missingness.analyses.day_influence
```

This slice changes import topology only. Estimating equations, constants, outputs, privacy guarantees, figures, and scientific conclusions are unchanged.

Legacy top-level module names remain available as compatibility entry points for notebooks, tests, workflows, and external links during the staged migration.
