# MCP Is Not Just Tool Calling

**Status:** draft  
**Theme:** AI engineering, interfaces, production systems

A surprising amount of discussion around Model Context Protocol reduces the idea to a code snippet that exposes a function to a language model.

That is the least interesting part.

The engineering value of MCP is the boundary it creates between a probabilistic model and the systems it is allowed to interact with.

That boundary deserves the same care as any other production interface.

## The model should not own the system contract

Suppose an AI assistant needs account data before generating a briefing.

One design gives the model access to a generic database query tool. Another exposes a narrow capability such as

```text
get_account_briefing_data(account_id)
```

The second interface gives the model less freedom, but the system becomes easier to reason about.

The application can define:

- exactly which fields may be returned;
- which identities are authorized to access them;
- how inputs are validated;
- what happens when upstream services fail;
- what gets logged;
- which version of the contract is in use.

This is a systems-design problem before it is a prompting problem.

## A language model is not an authorization layer

The model can decide that a tool appears relevant. It should not decide whether a user is permitted to execute the underlying operation.

Authorization belongs outside the model.

The useful separation is:

\[
\text{model intent}
\rightarrow
\text{typed request}
\rightarrow
\text{policy check}
\rightarrow
\text{execution}
\rightarrow
\text{validated result}.
\]

The model participates in the first step. It does not replace the others.

## Narrow tools are easier to evaluate

A broad interface pushes complexity into model behaviour.

A narrow interface pushes more responsibility into ordinary software, where deterministic tests are available.

That changes the evaluation problem.

Instead of asking only whether the model chose the right tool, we can test:

- whether the schema rejects malformed input;
- whether unauthorized requests fail independently of model output;
- whether retries are bounded;
- whether errors are represented consistently;
- whether responses respect size and privacy constraints;
- whether the tool contract changes compatibly across versions.

The less ambiguous the interface, the less work the model must perform implicitly.

## Observability matters more once tools can act

A text-generation failure can produce a bad paragraph.

A tool-using system can produce a side effect.

That changes the operational standard.

For each invocation we may need to know:

\[
\text{who requested it},
\quad
\text{what was requested},
\quad
\text{which policy allowed it},
\quad
\text{what executed},
\quad
\text{what came back}.
\]

This does not mean logging every private payload indiscriminately. It means designing an audit model deliberately.

## Failure is part of the interface

Production integrations fail.

Databases time out. APIs return partial data. Permissions change. Schemas evolve. Rate limits appear at inconvenient moments.

An MCP server therefore needs more than a happy-path function signature. It needs a failure contract the host application can interpret.

For example, there is an important difference between:

```text
account not found
```

and

```text
account service temporarily unavailable
```

If both become generic natural-language errors, the model is forced to infer operational state from prose.

Typed failures are usually better.

## MCP does not remove architecture

A standardized protocol can reduce integration friction. It does not remove the need to decide:

- where trust boundaries sit;
- which capabilities are exposed;
- which operations are read-only;
- which actions require confirmation;
- how credentials are isolated;
- what context the model should receive;
- how outputs are validated before they reach downstream systems.

Those are application architecture decisions.

## The useful mental model

I think of MCP less as "a way for an LLM to call functions" and more as

\[
\boxed{
\text{a standardized capability boundary around external systems.}
}
\]

That framing changes the implementation questions.

The first question stops being "how quickly can I expose this API?"

It becomes:

\[
\boxed{
\text{What is the smallest safe capability the AI application actually needs?}
}
\]

That is a much better place to start.
