# Escalation Policy — SpotifyCares Support Agent

## Ground-truth policy (used for golden-set labelling)

This is the policy a careful human support lead would apply given just the
customer message + conversation context + intent (no live account access).
It is the target the automated policy (`src/support_agent/escalation.py`,
Phase 10) is evaluated against. The automated policy approximates the same
judgment using signals actually available at inference time: predicted
intent, classifier confidence, retrieval similarity/evidence quality, and a
few keyword-based risk flags (see below) - it cannot see "account-specific
investigation needed" directly, so it has to infer that from proxies. The
gap between the two is expected and is reported in the ACTION/ESCALATION
evaluation (Phase 13) and failure analysis (Phase 14), not hidden.

**Default: AUTO_HANDLE.** We escalate only when there's a concrete reason,
because a system that escalates everything provides no value - the goal is
*safe automation*, not maximum escalation.

### Escalate when any of these hold:

1. **SENSITIVE_FINANCIAL** — the ask involves money moving (refund,
   chargeback, disputed/duplicate charge) in a way that requires looking at
   the customer's specific billing record or authorizing an action a
   generic reply can't authorize. Applies to essentially all
   `REFUND_REQUEST` cases and to `PAYMENT_BILLING_ISSUE` cases where the
   customer reports a *specific* wrong charge (not a general "how does
   billing work" question).
2. **ACCOUNT_SPECIFIC_INVESTIGATION** — resolving this requires looking at
   this customer's actual account state (e.g., "I was upgraded to Family
   without consent," "I followed the steps and still can't add my wife to
   the family plan," "my student verification has been pending 3 days") -
   a generic troubleshooting reply would be either wrong or unhelpfully
   vague.
3. **SECURITY_CONCERN** — account compromise/hacking, or anything requiring
   identity verification before acting.
4. **AMBIGUOUS_OR_INSUFFICIENT_INFO** — the message is too vague, mixes
   unrelated issues without a dominant one, or lacks enough detail to
   answer safely. Most `OTHER_AMBIGUOUS`-intent messages land here.
5. **REPEATED_UNRESOLVED_ISSUE** — the customer says they already tried the
   obvious fix, already emailed/DMed support, or this is a recurrence of a
   previously "resolved" issue. Generic advice would just repeat what
   already failed.
6. **OUT_OF_SCOPE** — outside what a generic support agent should touch
   (label/artist-relations disputes, unsolicited pitches, requests that
   aren't really a personal support need).

### Auto-handle when:

- The intent has a stable, generic, correct answer that doesn't depend on
  *this* customer's account state: standard troubleshooting steps for a
  common `TECHNICAL_PLAYBACK_ISSUE` (restart app / log out and back in /
  check for updates), general `FEATURE_REQUEST_OR_INFO` answers, general
  `CONTENT_CATALOG_AVAILABILITY` explanations (licensing/region limits),
  plain `SUBSCRIPTION_PLAN_MANAGEMENT` how-it-works questions, and
  `POSITIVE_FEEDBACK_OR_RESOLVED` acknowledgements.
- None of the six escalation triggers above apply.

### Note on `ACCOUNT_LOGIN_ACCESS`

Split by sub-case: a plain "I forgot my password / how do I reset it"
auto-handles (generic reset-link instructions are correct and safe). "My
account was hacked" or "I lost access and the recovery email is also gone"
escalates (SECURITY_CONCERN / ACCOUNT_SPECIFIC_INVESTIGATION). This is the
single intent where the *sub-case* matters more than the intent label
itself for the escalation decision — flagged explicitly in the annotation
notes for every `ACCOUNT_LOGIN_ACCESS` golden example.

## Automated policy signals (Phase 10 implementation)

The runtime policy cannot read the human's mind, so it escalates based on:

- **Intent-based prior risk**: `REFUND_REQUEST` always escalates regardless
  of confidence; a small set of "structurally risky" intents
  (`PAYMENT_BILLING_ISSUE`, `ACCOUNT_LOGIN_ACCESS`) get a lower
  auto-handle confidence threshold.
- **Classifier confidence** below a threshold (calibrated against the
  golden set, not guessed) → escalate as `LOW_CONFIDENCE`.
- **Retrieval evidence quality**: if the top retrieved historical
  resolutions have low similarity or disagree with each other
  (conflicting resolutions), escalate as `WEAK_OR_CONFLICTING_EVIDENCE`
  rather than let the generator guess/hallucinate.
- **Keyword risk flags**: presence of words like "hacked," "fraud,"
  "unauthorized," "lawyer," "cancel and refund," etc. force escalation
  regardless of predicted intent, as a safety net against misclassification
  on the highest-stakes cases.

`LOW_CONFIDENCE` and `WEAK_OR_CONFLICTING_EVIDENCE` are system-level reasons
with no human-labelled equivalent (a human doesn't have a "confidence
score") - they only appear in the automated policy's output, not in golden
labels. When comparing automated decisions to golden ground truth, these map
loosely onto whichever human reason category is the closest fit.
