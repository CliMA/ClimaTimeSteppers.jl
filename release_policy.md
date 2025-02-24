# Release policy

ClimaTimesteppers.jl is a core component to the CliMA ecosystem, and can have
a significant impact on users since most experiments and examples rely on this
infrastructure.

As such, we decided to enforce a release policy:

Before making breaking releases, a stable patch release must be exercised by
user repos and without new issues for at lease one week.

This will help users to avoid issues when updates, and developers to avoid the
need to create patch releases along old minor versions.
