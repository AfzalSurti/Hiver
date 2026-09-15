# Intent Taxonomy — SpotifyCares

Derived from reading ~150 sampled customer opening messages across
`data/processed/conversations.jsonl` (see `scripts/03_build_conversations.py`
for how these were built). 9 intents — slightly above the "6-10" target
because Spotify's support volume splits cleanly into a financial cluster
(billing / refunds / plan management are genuinely different asks with
different escalation implications) and a content cluster (technical bugs vs.
catalog/licensing gaps are frequently confused by customers but require
completely different resolutions), and collapsing either pair would blur an
escalation-relevant distinction.

Each intent lists: description, inclusion criteria, exclusion criteria,
2-3 real examples (verbatim from the dataset, cleaned), and known ambiguous
cases.

---

## 1. ACCOUNT_LOGIN_ACCESS

**Description:** Customer cannot log in, reset a password, recover, or
verify ownership of their account, including third-party (Facebook) login
links breaking.

**Inclusion:** login failures, password reset requests/errors, "forgot my
login info", account recovery after losing access, hacked/compromised
account, broken Facebook/social login linkage, CSRF/auth errors during
sign-in.

**Exclusion:** payment-method update failures (→ PAYMENT_BILLING_ISSUE even
though it happens "in account settings"); forgetting *which plan* they're on
(→ SUBSCRIPTION_PLAN_MANAGEMENT).

**Examples:**
- "help! I have Spotify logged in on my phone. I would like to log in on my computer as well. I can't remember my log in info!"
- "I'm getting a csrf token error when I try to sign in on my PC desktop or reset my password on my PC desktop."
- "My partner's was hacked - what's the fastest way to get it back?"

**Ambiguous cases:** "I closed my facebook account a year ago that was hooked up to my spotify now i cant get it back" — could be ACCOUNT_LOGIN_ACCESS (can't log in) or a third-party linkage issue; classified here because the end symptom is losing access to the account.

---

## 2. PAYMENT_BILLING_ISSUE

**Description:** Something about how the customer is being charged is wrong
or confusing, and they are not explicitly asking for money back.

**Inclusion:** payment method won't update / is rejected, charged an
unexpected amount, charged despite a trial/discount that should have
applied, confusion about being billed monthly *and* annually, promo/discount
codes not applying at checkout.

**Exclusion:** an explicit ask to cancel-and-get-money-back (→
REFUND_REQUEST); plan upgrade/downgrade mechanics with no billing error (→
SUBSCRIPTION_PLAN_MANAGEMENT).

**Examples:**
- "I am trying to update my payment details but 2 different cards from 2 different banks are 'having a bad day' but only with you"
- "I've been charged full price when I have the student offer"
- "Hi. I converted to family premium from single premium. Paid annual fee but looks like I will still be charged monthly as well?"

**Ambiguous cases:** billing confusion that ends with "so I want to cancel" blurs into REFUND_REQUEST — labelled REFUND_REQUEST only if cancellation/refund is the explicit primary ask, otherwise PAYMENT_BILLING_ISSUE.

---

## 3. REFUND_REQUEST

**Description:** Customer explicitly asks to cancel a paid subscription and/or
get money back, or disputes a charge as fraudulent/unauthorized and wants it
reversed.

**Inclusion:** "can I get a refund", cancelling mid-cycle and wanting a
pro-rated refund, disputing/wanting to reverse a specific charge, a
long-running unauthorized recurring charge the customer wants stopped and
reimbursed.

**Exclusion:** a plain cancellation with no mention of money back (→
SUBSCRIPTION_PLAN_MANAGEMENT); general billing confusion with no explicit
refund ask (→ PAYMENT_BILLING_ISSUE).

**Examples:**
- "Just canceled my premium subscription and not interested in using the rest of the month. Any way I can get a refund?"
- "my biz credit card gets charged every month but I don't know my account so can't cancel. It's been like 3 years. ... How do I cancel???" (long-running unwanted charge + explicit ask to stop it)

**Ambiguous cases:** this intent is inherently high-stakes/financial — see `docs/escalation_policy.md`, essentially all REFUND_REQUEST cases escalate regardless of confidence, since account-specific billing history is required to act and Spotify's historical replies to these threads are themselves usually "please DM us" (i.e., the brand also couldn't resolve it in-thread).

---

## 4. SUBSCRIPTION_PLAN_MANAGEMENT

**Description:** Questions or problems about which plan/tier the customer is
on, moving between plans, family/student plan membership, or cancelling
without asking for a refund.

**Inclusion:** upgrade/downgrade questions, adding/removing a family-plan
member, student-plan eligibility verification, being switched to a plan
without consent, plain cancellation requests (no refund ask), general "how
does the plan work" questions.

**Exclusion:** the money side of a plan (mischarged, wrong amount) →
PAYMENT_BILLING_ISSUE; wanting money back → REFUND_REQUEST.

**Examples:**
- "Like a whole bunch of other people, I can't add a new member (my wife) to my family plan."
- "I was just informed that I was upgraded to the Family plan (not by choice). I did not do this. Why would this happen?"
- "Sent in proof of my Student Enrollment on Sat 11/4. Said it would take an hr and still haven't heard back"

**Ambiguous cases:** student-discount *pricing* complaints ("why do I need a credit card for a $4.99 plan") sit at the boundary with PAYMENT_BILLING_ISSUE — classified SUBSCRIPTION_PLAN_MANAGEMENT when the core complaint is about plan eligibility/terms, PAYMENT_BILLING_ISSUE when it's about an actual charge already made.

---

## 5. TECHNICAL_PLAYBACK_ISSUE

**Description:** The app or a device integration is broken: something that
should work mechanically doesn't.

**Inclusion:** playback errors/crashes, offline downloads disappearing or
failing, app hanging/not launching, search not working, media-key/OS
integration not working (Control Center, keyboard buttons, Siri, Android
Auto), sync/timeout issues across devices, in-app bugs (playlists
disappearing due to a bug, not user action), ads misbehaving on free tier.

**Exclusion:** a song/album/artist being unavailable because Spotify doesn't
have the licensing rights (→ CONTENT_CATALOG_AVAILABILITY) — the distinction
customers themselves usually can't make, but the resolutions differ
completely (bug fix vs. "we don't have the rights, no ETA").

**Examples:**
- "My spotify constantly deletes all my saved music. I reported yesterday & got no reply. Happened again today."
- "I'm having issues on iOS11 with my iPhone 8 Plus not resuming music after a phone call, or using Siri."
- "why does my Spotify stop playing while offline? It is songs I've downloaded and haven't had the issue until yesterday."

**Ambiguous cases:** "just heads up that Kendrick Lamar tracks still aren't working. Sometimes I can listen to them and then 15 minutes later, error" — could be a playback bug or a licensing/catalog hiccup for that specific catalog; classified TECHNICAL_PLAYBACK_ISSUE because it's intermittent (bug-shaped), not a flat "unavailable."

---

## 6. CONTENT_CATALOG_AVAILABILITY

**Description:** A specific song, album, artist, or podcast is missing,
region-locked, mislabeled, or the customer wants something added to the
catalog.

**Inclusion:** "why isn't X available", a track/album removed from the
catalog, wrong artist/song metadata, region availability questions, requests
to add specific artists/podcasts to the catalog.

**Exclusion:** intermittent playback errors on content that mostly works (→
TECHNICAL_PLAYBACK_ISSUE).

**Examples:**
- "My track seems to have been taken down from Spotify for some reason. How can I find out what has happened?"
- "is there a particular reason why 'Kashmir' is missing from LZ's Mothership?!?"
- "THIS is the song...Fad.e by Frankie P But this guy is not her Different artists. She needs a new profile" (wrong metadata)

**Ambiguous cases:** artist/label disputes over misattributed songs shade into "customer is actually the artist" cases (rare, out of scope for a generic support agent — these get OTHER_AMBIGUOUS since they need specialist/label-relations handling, not standard support).

---

## 7. FEATURE_REQUEST_OR_INFO

**Description:** The customer isn't reporting something broken; they're
asking how something works, asking for a feature that doesn't exist, or
asking a general informational question.

**Inclusion:** "how do I...", feature requests (rearrange playlists, exclude
explicit content, block an artist), general product questions (regional
availability of the service itself, gift cards, promotions, "will there be a
Year in Review"), account-adjacent info questions that aren't a problem
report.

**Exclusion:** a "how do I" that's actually someone stuck on a real error (→
TECHNICAL_PLAYBACK_ISSUE or ACCOUNT_LOGIN_ACCESS depending on the error).

**Examples:**
- "When are you going to have 'show already played tracks' in Spotify? It is very hard to keep up with playlists!"
- "can you purchase gift cards from best buy or walmart?"
- "Hi, will there be a Year in Music 2017?"

**Ambiguous cases:** "Is there a setting to block any and all Chris Brown from appearing in my Spotify?" is a feature request (no such setting) but reads like a complaint — classified FEATURE_REQUEST_OR_INFO since there's no bug and no billing/account impact, it's "does X exist."

---

## 8. POSITIVE_FEEDBACK_OR_RESOLVED

**Description:** The customer is thanking the brand, confirming their issue
is already resolved, or giving unsolicited praise — no action is needed.

**Inclusion:** "thanks, fixed now", praise for a specific support agent,
general positive sentiment with no open ask.

**Exclusion:** praise followed by a *new* ask (classify by the new ask
instead).

**Examples:**
- "problem fixed. thank you so much for addressing it"
- "Big up to Saira L who I dealt with online today. Great service and speedy responses: always nice to see."

**Ambiguous cases:** none observed that weren't clear-cut; this is the
lowest-ambiguity intent in the taxonomy.

---

## 9. OTHER_AMBIGUOUS

**Description:** Catch-all for messages that are too vague to act on, are
off-topic, mix multiple unrelated issues so badly that a single intent label
would mislead, or fall outside what a generic support agent should handle
(e.g., label/artist-relations requests, unsolicited pitches to the brand).

**Inclusion:** single-word or near-content-free messages ("i have a
question"), messages requesting a callback/DM with zero description of the
underlying issue, unsolicited suggestions/pitches unrelated to a personal
support need, multi-issue rambles where no single intent dominates.

**Exclusion:** anything that fits cleanly into 1-8, even if short — brevity
alone doesn't make something OTHER_AMBIGUOUS (e.g. "can I get a refund?" is
short but clearly REFUND_REQUEST).

**Examples:**
- "i have a question"
- "Hey , here's a quick idea on how you can use your platform to help those affected by tragedies..." (unsolicited pitch, not a support need)
- "Answer my DM! Urgent, thanks." (no content about the underlying issue at all)

**Ambiguous cases:** this entire intent *is* the ambiguous bucket by
design — the judgment call is whether enough signal exists to pick 1-8
instead. The golden-set annotation notes (`data/golden/ANNOTATION_METHODOLOGY.md`)
record this decision explicitly for every OTHER_AMBIGUOUS example.

---

## Notes on overlap and design choice

- PAYMENT_BILLING_ISSUE vs. REFUND_REQUEST vs. SUBSCRIPTION_PLAN_MANAGEMENT
  form a deliberately fine-grained financial cluster: an LLM/human collapsing
  them would lose exactly the distinction the escalation policy needs (a
  refund ask escalates far more aggressively than a plan question).
- TECHNICAL_PLAYBACK_ISSUE vs. CONTENT_CATALOG_AVAILABILITY are the hardest
  pair for both humans and models — customers describe both as "X doesn't
  work" — and are kept separate anyway because their correct resolutions are
  unrelated (bug report/troubleshooting steps vs. "we don't have the
  licensing rights"). Expect this pair to dominate the confusion matrix; see
  `REPORT.md` failure analysis once evaluation is run.
