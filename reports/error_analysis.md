# Systematic Error Analysis: Active Champion vs Validation Partition

**Validation Population**: 20,000 Frozen S1 Entities (`data/frozen_val_s1_ids.json`)

## 1. Quantitative Breakdown of Errors

| Error Mode | Count | Root Cause | Architectural Mitigation |
| :--- | :--- | :--- | :--- |
| **Singleton False Merges** | 115 | Loose similarity threshold matches distractor on singletons | Strict score margin guard & top-1 ambiguity check |
| **Retrieval Upper Bound Gap** | 7,260 | True targets not retrieved by simple token/street indexing | Character n-grams (3/4-gram) + rare token channels |
| **End-to-End Missed Matches** | 11,314 | Scoring below tau=0.56 due to heavy address variation | Hybrid ML reranker for ambiguous medium-score pairs |
| **Same Name / Different Location** | 850 | Brand/chain stores across different cities/localities | Locality / Postal code agreement filter |
| **Shared Address Number False Positives** | 1 | Distinct co-located tenants at same street address | Heavier name agreement requirement when address matches |

## 2. Sample False Merges on Singletons (Examined)

- **S1**: `S1-218685314` | Name: *ortem future of hyderabad pvt ltd* | Addr: *pt no 394 satavahana nagar kukatpally hyderabad telangana*
  **Cand**: `S3-172380390` | Name: *pvt ortem future of hyhdeirbd ventures ltd* | Addr: *tg hyderabad pt no 398 satavahana nagar ranga ready kukatpally*
  *Score*: 0.6348 (False merge — S1 is actually a singleton)

- **S1**: `S1-14671216` | Name: *shiv constructions private limited* | Addr: *s no 27 8b 1st floor flat no 1 ambegaon bk nr toramkar fmill pune maharashtra*
  **Cand**: `S3-551214339` | Name: *shiv constructions trading* | Addr: *1st floor flat no 1 ambegaon bk nr toramkar fmill pune s no 27 29b mh*
  *Score*: 0.7457 (False merge — S1 is actually a singleton)

- **S1**: `S1-910195628` | Name: *office of aging* | Addr: *107 portland street morristown vt*
  **Cand**: `S3-584191562` | Name: *office of aging llc* | Addr: *charlotte north carolina 5006 evanders way unit 304*
  *Score*: 0.6064 (False merge — S1 is actually a singleton)

- **S1**: `S1-910195628` | Name: *office of aging* | Addr: *107 portland street morristown vt*
  **Cand**: `S2-139219752` | Name: *office of aging inc* | Addr: **
  *Score*: 0.6064 (False merge — S1 is actually a singleton)

- **S1**: `S1-736903406` | Name: *step law chambers private limited* | Addr: *hn 82 plot no 404 ward no 32 gaya bihar*
  **Cand**: `S3-161815031` | Name: *shri step law chambers plrviae limited* | Addr: *h no 29 ground floor sri ambal nagar main road tambaram kanchipuram tn*
  *Score*: 0.5703 (False merge — S1 is actually a singleton)

## 3. Sample Retrieval Bottlenecks (True targets not in candidates)

- **S1**: `S1-27541239` | Name: *nexus anchor rain* | Addr: *1111 church street unit 2007 nashville tn*
  *Missing True Targets*: ['S3-893131105', 'S2-85756610', 'S3-739757636']

- **S1**: `S1-865131206` | Name: *helios* | Addr: *66 edgewood street bridgeport ct*
  *Missing True Targets*: ['S3-652513823', 'S2-657355663']

- **S1**: `S1-626914593` | Name: *unified choice dynamix* | Addr: *unit building 3030 md 2701 eastern boulevard middle river*
  *Missing True Targets*: ['S2-563003232']

- **S1**: `S1-252242682` | Name: *aggie e nagle l c s w* | Addr: *11900 54th avenue plymouth mn*
  *Missing True Targets*: ['S2-84842746']

- **S1**: `S1-883973644` | Name: *global hovnanian llc* | Addr: *4024 39th avenue seattle wa*
  *Missing True Targets*: ['S2-346262429']

