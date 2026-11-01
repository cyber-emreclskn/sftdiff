# Contributing

Bug reports, small focused improvements, and reproducible examples are welcome.
Discuss substantial feature changes in an issue before implementation. Use
synthetic examples instead of private training data.

## Development

Use Python 3.10 or newer and Node.js 22 for the report UI tests:

```sh
python -m pip install -e .
python -m unittest discover -s tests -v
node --test tests/test_report_ui.cjs
sftdiff demo --out demo-run
```

The core package has no runtime dependencies. Keep optional integrations optional.
Add regression coverage for behavior changes and describe how you verified them.

## Pull requests

Fork the repository, create a branch, and open a pull request against `main`.
Describe the problem, resulting behavior, and validation. All four CI jobs must
pass against the latest `main`, and review conversations must be resolved.
Contributions require a code-owner approval; new commits dismiss old approvals.

This repository currently has one maintainer. Repository admins can bypass the
review-approval rule through a pull request so they can merge their own work.
A separate rule still requires a pull request and passing CI for everyone,
including admins. Force pushes and deletion of `main` are blocked. Changes are
squash-merged and merged branches are automatically deleted.

CI uses read-only repository permissions. Workflows from external contributors
require approval before running. Do not introduce `pull_request_target` workflows
that execute untrusted pull-request code. Pin third-party actions to full commit
SHAs; Dependabot proposes updates.

For security vulnerabilities, follow [SECURITY.md](SECURITY.md).
