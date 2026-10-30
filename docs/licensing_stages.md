# Review and public-release stages

## Current review stage

The root LICENSE grants limited permission for authorized confidential manuscript
review, including local execution and verification. It is not an open-source
license. Third-party materials retain their original licenses without additional
restrictions on the rights those licenses grant. Both paper implementations in
this checkout follow this distinction; publication of the earlier paper does not
by itself assign a license to this software package.

Distribute only the intended review snapshot through the journal or an agreed
private channel. Do not include the private Git history or inactive MIT document
backups in a review package. Earlier local commits contain MIT wording; changing
the current text does not revoke rights already validly granted to any recipient.
No public distribution of those commits is asserted here.

## Planned post-acceptance stage

1. Obtain the copyright holders' authorization to issue the public MIT version.
2. Replace the root review LICENSE with the saved MIT text; confirm copyright
   names and years. Keep all third-party copyright and license notices.
3. Update README, THIRD_PARTY_NOTICES.md and CITATION.md consistently. Remove
   review-only wording and add the confirmed paper citation and release metadata.
   Preserve technical documentation changes made since the saved snapshot.
4. Check packaging metadata and the actual distribution contents, commit and tag
   the MIT release, and publish only when authorized.

Acceptance does not automatically relicense a review copy; the MIT grant must be
included explicitly in the separately issued release. The saved MIT-stage
LICENSE, README, THIRD_PARTY_NOTICES.md and CITATION.md are held outside the
repository in the maintainer's private archive, with a checksum manifest and
restoration instructions. They are templates, not the active review terms.
