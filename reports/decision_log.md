# Decision Log

**Author: Shivendra Kirar**

Non-obvious decisions made during this project, with rationale. Ordered
roughly by when they came up.

**1. Target brand: AmazonHelp, not a brand named "Hiver"**
The assignment is from a company called Hiver, but the dataset itself only
contains real brand support accounts (AmazonHelp, AppleSupport, etc.). I chose
AmazonHelp for volume and recognizability.

**2. Separated intent classification from escalation decision-making**
I initially considered a `general_complaint_or_escalation` intent class. I
rejected this: it would leak the escalation answer into the intent label (an
angry, repeated complaint about a late package is still fundamentally about
the delivery, not a new topic). I modeled escalation as an independent
decision layered on top of intent, with its own reason codes.

**3. Filtering rule: signal-based, not length-based**
I initially planned to filter out short messages (<15 chars) before
classification. I rejected this in favor of a signal-based filter: "Refund?"
and "Wrong item." are short but fully informative; length alone is an
arbitrary and misleading proxy for whether a message is worth classifying.

**4. `language_out_of_scope` kept distinct from `non_support_noise`**
Non-English messages are not noise — many are genuine, urgent support
requests (one example in the raw data was an angry, non-English message
about money being stuck, clearly a real complaint). Conflating the two would
mean silently mislabeling real support needs as garbage, so I kept them as
separate labels.

**5. `subscription_or_membership_issue` folded into `other_support_issue`**
I measured this at only 1.0% (10/1000) in the classified sample — too thin
to justify as a standalone class in a ~200-example golden set, where it
would get at most 1-2 examples. I folded it in rather than force-split a
near-empty category.

**6. Switched classification backend twice: Claude API -> Gemini free tier ->
local Ollama**
Claude API required paid credit; Gemini's free tier proved unusable at scale
(20 requests/day limit, which I confirmed via the AI Studio usage dashboard —
it would have taken weeks for 1000 classifications). I settled on local
Ollama (Llama 3.1): zero cost, no rate limits, fully reproducible without
any API key.

**7. Chose not to re-engineer thread reconstruction after measuring an 11.6%
fragment-as-opener rate**
I diagnosed that "first customer turn" extraction sometimes grabs a
mid-conversation reply rather than a true thread opener (measured at 11.6%
across all 82,541 threads). The fix (requiring a true root tweet, with no
`in_response_to_tweet_id`) would silently drop threads rather than clearly
improve accuracy. I documented this as a measured, known limitation instead
of patching it under time pressure.

**8. Rejected a fabricated 200-example "99.5% agreement" golden set, at the
cost of falling short of the 150-250 spec**
An initial labeling pass I did was rushed — I accepted suggested labels
without reading the messages. Rather than report that number, I redid it as
an honest, carefully-reviewed 44-example stratified spot-check (75.0% real
agreement). I judged a smaller, trustworthy number to be more valuable than
a larger, fake one — but 44 is short of the 150-250 examples this assignment
specifies, and that gap is real, not resolved by the honesty of the smaller
check. I've listed it as the top item in Next Steps.

**9. Reply generation explicitly forbidden from inventing URLs or specific
commitments**
Early test generations fabricated a plausible-looking but fake tracking URL.
I added an explicit prompt rule: never include a URL unless copied verbatim
from the customer's message or a retrieved real brand reply. I applied the
same principle to refund amounts, dates, and compensation promises.

**10. Retrieval index excludes the query's own thread**
I found that the RAG retriever's top "similar" match was frequently the
query message's own historical thread (similarity=1.0) when the golden set
overlaps with the index it searches — a direct data leak. I added an
`exclude_thread_id` parameter to prevent the generator from trivially
"grounding" a reply by looking up its own answer.

**11. LLM-as-judge scores were investigated before being trusted, not
accepted at face value**
Near-perfect reply-quality scores (4.84-5.0/5 across all 44 examples and 11
intent classes) were suspicious on their face — the same failure pattern as
decision #8. I manually inspected the judge's rationales and confirmed they
were specific and discriminating (the judge correctly caught the one genuine
miss in the sample), and concluded the scores reflect a genuinely easy
sub-task pattern rather than a broken or lenient judge. I've documented
same-model-judges-itself as a residual limitation regardless.

**12. Two baselines chosen specifically to be "honest floors," not straw men**
I built the keyword-rule baseline to actually work on obvious cases (verified
against 9 hand-picked test cases before running it for real) rather than
making it deliberately weak — so its 20.5% result against the LLM's 75.0% is
a fair, defensible comparison rather than an inflated contrast.

**13. Report written in Markdown, not Word, for the primary deliverable**
Given the strict page-length constraint and my need to iterate quickly, I
chose Markdown for the working report rather than docx — easier to review
and revise; can be converted to PDF/Word if the submission format requires
it.
