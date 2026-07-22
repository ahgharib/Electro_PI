# Test Questions — NIST 800-145 / NASA Solar System / CDC Flu VIS

15 questions against the 3 uploaded PDFs. Every reference answer was
checked directly against the extracted PDF text (the same text your
`PdfLoader` actually produces — verified above, not assumed). Because the
three documents share no topic overlap, there are no cross-document
synthesis questions this round — see the chat message for why, and how to
add that category back if you want it.

Legend: 🟢 easy · 🟡 medium · 🔴 hard · ⚫ edge case (no relevant context / grounding test)

---

## NIST SP 800-145 — The NIST Definition of Cloud Computing

**Q1 🟢 How many essential characteristics, service models, and deployment models does the NIST definition of cloud computing have?**
Ground truth: 5 essential characteristics, 3 service models, 4 deployment models.

**Q2 🟢 Who are the authors of NIST Special Publication 800-145, and when was it published?**
Ground truth: Peter Mell and Timothy Grance; September 2011.

**Q3 🟡 What are the three cloud service models, and which one gives the consumer the least control over the underlying infrastructure?**
Ground truth: SaaS, PaaS, IaaS. SaaS gives the least control — the consumer only uses the provider's applications, with at most limited user-specific configuration settings; they don't manage network, servers, OS, storage, or even individual application capabilities.

**Q4 🟡 According to the document, what distinguishes a "community cloud" from a "private cloud"?**
Ground truth: A private cloud is provisioned for exclusive use by a single organization (possibly multiple business units within it). A community cloud is provisioned for exclusive use by a specific community of consumers from *multiple* organizations that share concerns such as mission, security requirements, policy, and compliance considerations.

**Q5 🔴 Under which specific law and legal authority did NIST develop this document, and does the document claim copyright?**
Ground truth: Developed under the Federal Information Security Management Act (FISMA) of 2002, Public Law 107-347. The document explicitly states it "is not subject to copyright, though attribution is desired."
*Why it's hard*: this is a low-salience legal/administrative detail buried in section 1.1, not the "main content" — a good test of whether retrieval favors the popular/central definition chunks over this correct-but-easy-to-miss one.

**Q6 🔴 Which essential characteristic involves a "multi-tenant model," and what example does the document give for how much location control a consumer might have?**
Ground truth: Resource pooling. The document notes the consumer generally has no control or knowledge of the exact resource location, but may be able to specify location at a higher level of abstraction — e.g. country, state, or datacenter.

---

## NASA "Our Solar System" lithograph

**Q7 🟢 Which planets are classified as terrestrial planets, and which are classified as gas giants vs. ice giants, in this document?**
Ground truth: Terrestrial: Mercury, Venus, Earth, Mars. Gas giants: Jupiter, Saturn. Ice giants: Uranus, Neptune.

**Q8 🟢 According to this document, roughly how old is our solar system, and what is one Astronomical Unit (AU) in kilometers?**
Ground truth: About 4.6 billion years old. 1 AU ≈ 150 million kilometers (93 million miles) — the Earth–Sun distance.

**Q9 🟡 According to the FAST FACTS table, how many moons does Neptune have, and does the document note any caveat about that number?**
Ground truth: 13 known moons per the table, with a footnote (∆) that Neptune has 1 additional moon awaiting official confirmation, bringing the total to 14.

**Q10 🔴 As of this document, how far had Voyager 1 and Voyager 2 traveled from the Sun, and in what years did each cross the "termination shock"?**
Ground truth: By 2013, Voyager 1 was about 18 billion km (11 billion miles) from the Sun; Voyager 2 was about 15 billion km (9 billion miles). Voyager 1 crossed the termination shock in 2004, Voyager 2 in 2007. (Both spacecraft launched in 1977.)

**Q11 🔴 Is the moon and distance data in this document current, and how should an answer handle that?**
Ground truth-shaped test, not a single fact: the document itself says its moon counts are "known moons as of July 2013" and flags several as provisionally unconfirmed at the time (e.g. "Jupiter has 17 moons awaiting official confirmation, bringing the total to 67"). A correct, well-grounded answer should present the 2013 figures as *what this document reports* rather than asserting them as today's real moon counts (which are now higher for every giant planet) — and should ideally flag the document's own "as of July 2013" caveat.
*Why it's the most important question in this set*: it directly tests whether the pipeline is grounded in the retrieved document rather than silently substituting the model's own training-data knowledge — the single most important property this whole system exists to guarantee.

---

## CDC Vaccine Information Statement — Inactivated/Recombinant Influenza Vaccine

**Q12 🟢 How many doses of flu vaccine do children aged 6 months through 8 years need in a single flu season, versus everyone else?**
Ground truth: Children 6 months through 8 years of age may need 2 doses in a single flu season; everyone else needs only 1 dose per season.

**Q13 🟡 What is VAERS, and how do you report a reaction to it?**
Ground truth: VAERS = Vaccine Adverse Event Reporting System. Report via www.vaers.hhs.gov or by calling 1-800-822-7967. The document explicitly notes VAERS is only for reporting reactions, and VAERS staff do not give medical advice.

**Q14 🟡 What is the National Vaccine Injury Compensation Program, and is there a time limit to file a claim?**
Ground truth: VICP is a federal program created to compensate people who may have been injured by certain vaccines. Yes — claims for alleged injury or death due to vaccination have a filing time limit, which may be as short as two years. (Info: www.hrsa.gov/vaccinecompensation, 1-800-338-2382.)

**Q15 🔴 Under what circumstances does the document say a child getting a flu shot might have a slightly higher chance of a seizure, and what should you tell your provider?**
Ground truth: Young children who get the flu shot at the same time as the pneumococcal vaccine (PCV13) and/or DTaP vaccine might be slightly more likely to have a seizure caused by fever. Tell your health care provider if the child has ever had a seizure.

---

## ⚫ Edge cases — cross-topic, should not trigger "no relevant context"

Since the three documents share no topic overlap, asking a question whose subject belongs to one document against the *other two* is a clean, realistic no-context test — more realistic than an arbitrary off-topic question, since it checks that retrieval doesn't grab a superficially-plausible-sounding chunk from the wrong document just because a query uses generic language.

**Q16 According to these documents, how many doses of a cloud service does a small business need?**
Expected: no-context refusal — "doses" and "service" both appear in the corpus (flu doses, cloud/IaaS "service"), but not together in any real answer; this specifically tests against a false-positive keyword match across unrelated docs.

**Q17 What does NIST recommend for the flu vaccine schedule?**
Expected: no-context refusal — NIST appears only in the cloud computing document; nothing in that document mentions vaccines. Tests that the model doesn't conflate "NIST" as an authority figure across the whole corpus.

**Q18 How many moons does the cloud have, according to these documents?**
Expected: no-context refusal — deliberately absurd cross-topic phrasing ("moons" + "cloud") to confirm the gate doesn't get pulled toward the Solar System doc by the word "cloud" appearing nowhere near "moons" in it, nor toward the NIST doc by "cloud" appearing constantly there but never near "moons."

## ⚫ Edge cases — no relevant topic, should trigger "no relevant context"

Q23 — What is the maximum towing capacity of a diesel pickup truck?

Q24 — How do you file a small claims lawsuit in California?
---

## Note on verification

All ground-truth answers above were checked against the raw text your
`PdfLoader` actually extracts from each file (confirmed via a direct test
run, not assumed).
Retrieval *quality* and real generated answers still need a live run — Q1–Q15 are
ready to drop into `EVAL_QUESTIONS` in `scripts/eval.py`, or run one at a
time with `scripts/ask.py`.
