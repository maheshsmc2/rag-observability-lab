# Day 104 Metrics

## Answerable (golden_answerable.json)

- Total: 10
- Correct: 0
- Precision@1: 0.0
- Avg latency (ms): 302.92

## Unanswerable (golden_unanswerable.json)

- Total: 10
- Correct abstains: 4
- False confidence: 6
- Abstain accuracy: 0.4
- False confidence rate: 0.6
- Avg latency (ms): 27.58

## Failures for Day 105

### False confidence (expected ABSTAIN, got ANSWER): 6
- GU_02: What is the official HR email address? (got ANSWER)
- GU_03: What clause number defines medical leave during probation? (got ANSWER)
- GU_04: Who approved the attendance policy? (got ANSWER)
- GU_05: What is the penalty amount for late login? (got ANSWER)
- GU_06: Is work from home allowed during probation? (got ANSWER)
- GU_10: What happens if attendance is not marked for a day? (got ANSWER)

### False negatives (expected ANSWER, got ABSTAIN/UNKNOWN): 4
- GA_01: What is the probation period for permanent employees? (got ABSTAIN)
- GA_04: Does the notice period apply during probation? (got ABSTAIN)
- GA_06: Is attendance tracking mandatory for employees? (got ABSTAIN)
- GA_08: Is resignation allowed during the probation period? (got ABSTAIN)

### Wrong answered (expected ANSWER but incorrect): 6
- GA_02: Is medical leave allowed during the probation period?
- GA_03: How many days of notice are required for resignation?
- GA_05: Can an employee take leave during probation?
- GA_07: Does the company define different leave types?
- GA_09: Is approval required to take leave?
- GA_10: Does company policy mention probation conditions?
