# RESOLVE.AI: Empirical Error Analysis & Failure Mode Taxonomy

- **Evaluated Entities**: 20,000
- **Optimal Threshold**: $\tau^* = 0.69$
- **Macro F0.5**: 0.3973

## 1. Top Failure Modes Identified

1. **Locality-Level Sibling Businesses**: Businesses sharing common generic words (e.g. 'Apex Dental', 'Apex Auto') on the same commercial avenue.
2. **Address Street Number Transposition**: Inconsistent door numbering conventions in municipal databases.
3. **Cross-Source Legal Suffix Noise**: One source includes 'Pvt Ltd' while external records write 'India Solutions'.

## 2. Sample False Merges on Singletons (Critical Error Category)

| Source 1 Business Name | Source 1 Address | Matched Distractor Name | Distractor Address | Error Rationale |
| :--- | :--- | :--- | :--- | :--- |
| Little Diner Center | AL, Hodges, 541 117 | Flanigan Nevada, Group | 1056 HODGES ST, AMARILLO, TX | Sibling name distractor |
| Little Diner Center | AL, Hodges, 541 117 | PARRIS NANO UPTOWN LLC CENTER | 2913- HODGES STREET, AMARILLO, TX | Sibling name distractor |
| Little Diner Center | AL, Hodges, 541 117 | Group Great Strategic Emera | 4512-4516 HODGES RUN LANE, TX, HUMB | Sibling name distractor |
| Little Diner Center | AL, Hodges, 541 117 | J/U Ship | 151 HODGES LANE, WHITE STONE CITY,  | Sibling name distractor |
| Lakshmi Builders Private Limit | D-109, Gauri Shanker Enclave, Prem  | Kushinagar Ventures Private Lt | NO 933 97, SARAITAKI, 4JU160103 NEA | Sibling name distractor |
| Lakshmi Builders Private Limit | D-109, Gauri Shanker Enclave, Prem  | ALPHA INFOTECH ENTERPRISES PRA | SHANKER COMPLEX II FLOOR ##223 SULT | Sibling name distractor |
| Lakshmi Builders Private Limit | D-109, Gauri Shanker Enclave, Prem  | रॉयल सॉल्यूशंस एलएलपी | KASAM SULEMAN CHOHAN BLDG, MUMBAI C | Sibling name distractor |
| Lakshmi Builders Private Limit | D-109, Gauri Shanker Enclave, Prem  | Sri Anand Developers Private L | H.NO 312- OFFICE NO. 611, GAURI COM | Sibling name distractor |
| Lakshmi Builders Private Limit | D-109, Gauri Shanker Enclave, Prem  | MN Multi Industries [Private-L | HOUSE NO. G-431/5, GRAM-GAURI WARD- | Sibling name distractor |
| Lakshmi Builders Private Limit | D-109, Gauri Shanker Enclave, Prem  | TRANSWORLD INSTITUTE OF TECHNO | NO. 387 RZ-913, KH. NO. 910-12-13-1 | Sibling name distractor |
| Lakshmi Builders Private Limit | D-109, Gauri Shanker Enclave, Prem  | इंटरनेशनल टेक्नोलॉजी प्राइवेट  | PLOT NO.12., E-BLOCK, QUTAB VIHAR,  | Sibling name distractor |
| Lakshmi Builders Private Limit | D-109, Gauri Shanker Enclave, Prem  | श्री टेक्नोलॉजी लिमिटेड | DAYA SHANKER SHUKLA C/O BAISPUR VIJ | Sibling name distractor |
| Lakshmi Builders Private Limit | D-109, Gauri Shanker Enclave, Prem  | Preview Up LLP - 5045639133 | PLOT NO 237 SEMRA GAURI NEAR GOGNA  | Sibling name distractor |
| Lakshmi Builders Private Limit | D-109, Gauri Shanker Enclave, Prem  | Arrowline Trading Pvt | C/0 RAM SHANKER SINGH, VILL-RAGHOPU | Sibling name distractor |
| Lakshmi Builders Private Limit | D-109, Gauri Shanker Enclave, Prem  | TUF Private Límited Services | NO. 153 S NO. 2/2/3, DAMODHAR NAGAR | Sibling name distractor |

## 3. Sample Missed True Matches (False Negatives)

| Source 1 Business Name | Source 1 Address | Target True Match Name | Target True Match Address | Root Cause |
| :--- | :--- | :--- | :--- | :--- |
| Orelee's Barbershop | 1795 Westchester Drive, High Point, | Orelee'S Services | Westchester Dr, High Point, North C | Heavy abbreviation / missing address tokens |
| Prabhav Business Center | 797, Lake Town Block A, Kolkata, Ho | Prabhav Center Center | HN 567 797, LAKE TOWN BLOCK A, KOLK | Heavy abbreviation / missing address tokens |
| Prabhav Business Center | 797, Lake Town Block A, Kolkata, Ho | Mr Prabhav Business Services | Plot 334 797, Lake Town Block A, Ko | Heavy abbreviation / missing address tokens |
| Helios | 66 Edgewood Street, Bridgeport, CT | HELI0S LP | 66 EDGEWOOD ST, BRIDGEPORT, CT | Heavy abbreviation / missing address tokens |
| Helios | 66 Edgewood Street, Bridgeport, CT | Hiros | Connecticut, 66 Edgewood St, Bridge | Heavy abbreviation / missing address tokens |
| Aggie E. Nagle, L.C.S.W. | 11900 54th Avenue, Plymouth, MN | AGGIE E. NLMGL, (L.C.S.W.) | 54ND AVENUE, MINNEAPLIS, MN | Heavy abbreviation / missing address tokens |
| Aggie E. Nagle, L.C.S.W. | 11900 54th Avenue, Plymouth, MN | *** aggieenagle.com | 54ND AVENUE, MINNEAPLIS, MN | Heavy abbreviation / missing address tokens |
| Global Hovnanian LLC | 4024 39th Avenue, Seattle, WA | GLOBAL HOGNANINN LLC | 4024B 39RD AVE, SEATTLE, WA | Heavy abbreviation / missing address tokens |
| Pacific Learning Laboratories  | 4810 Nassau Avenue, Sand Springs, O | Pacific Learning Laboratories  |  | Heavy abbreviation / missing address tokens |
| George Saul Inc | Unit UNIT 367, 1400 Great Wolf Driv | Úmbrayuma | 1400 Great Wolf Dr, Unit UNIT 367,  | Heavy abbreviation / missing address tokens |
| George Saul Inc | Unit UNIT 367, 1400 Great Wolf Driv | georgesaul.com | 1400 Great Wolf Drive, Unit UNIT 36 | Heavy abbreviation / missing address tokens |
| Corner Hypnosis | 26 Stone Street, Unit 3, Beverly, M | Corner-Hypnosis | 26 Stone Street, PMB 5084, Beverly, | Heavy abbreviation / missing address tokens |
| Corner Hypnosis | 26 Stone Street, Unit 3, Beverly, M | Arclumjax Labs | 26 Stone Street, Unit 3, Beverly, M | Heavy abbreviation / missing address tokens |
| Corner Hypnosis | 26 Stone Street, Unit 3, Beverly, M | Brixlyra | PMB 9599, MA, BEVERLY, 0026 STONE S | Heavy abbreviation / missing address tokens |
| Corner Hypnosis | 26 Stone Street, Unit 3, Beverly, M | Cornerhypnosis.Com | 26 Stone Street, # 3, Beverly, Mass | Heavy abbreviation / missing address tokens |

## 4. Sample Confirmed Correct Matches (Hard Positives)

| Source 1 Business Name | Source 1 Address | Resolved Candidate Name | Resolved Candidate Address | Key Features |
| :--- | :--- | :--- | :--- | :--- |
| Orelee's Barbershop | 1795 Westchester Drive, High Point, | Orelee's (Barbershop) | #1795 Westchester Dr, North Carolin | Shared address tokens & high Jaro-Winkler |
| Orelee's Barbershop | 1795 Westchester Drive, High Point, | Orelee's Barbershop | 1795 WESTCHESTER DRIVE, NC, HIGH PO | Shared address tokens & high Jaro-Winkler |
| Orelee's Barbershop | 1795 Westchester Drive, High Point, | Orelee's Bárbershop | 1795 Westchester Drive, HIGH POINT, | Shared address tokens & high Jaro-Winkler |
| B+ Retail Inc | 1712 Montebello Avenue, Phoenix, AZ | B+ Rétail Incorporated | 1712 Montebello Ave, Northeast, Ari | Shared address tokens & high Jaro-Winkler |
| B+ Retail Inc | 1712 Montebello Avenue, Phoenix, AZ | B+ Retail | PHOENIX, AZ, 1712 MONTEBELLO AVE | Shared address tokens & high Jaro-Winkler |
| B+ Retail Inc | 1712 Montebello Avenue, Phoenix, AZ | B+ Inc Services | Arizona, Phoenix, 1712 Montebello A | Shared address tokens & high Jaro-Winkler |
| B+ Retail Inc | 1712 Montebello Avenue, Phoenix, AZ | B+ Retail INC (ID: 84923) | 1712 Montebello Avenue, Phoenix, Ar | Shared address tokens & high Jaro-Winkler |
| Nexus Anchor Rain | 1111 Church Street, Unit 2007, Nash | Nexus-Anchor Rain | NASHVILLE, TN, 1111 CHURCH STREET | Shared address tokens & high Jaro-Winkler |
| Nexus Anchor Rain | 1111 Church Street, Unit 2007, Nash | Nexus Anchor  Service | 1111 Church Street, Nashville, Tenn | Shared address tokens & high Jaro-Winkler |
| Nexus Anchor Rain | 1111 Church Street, Unit 2007, Nash | NEXUS ACORMHR RAIN | CHURCH STREET, NASHVILLE, TN | Shared address tokens & high Jaro-Winkler |
| Nexus Anchor Rain | 1111 Church Street, Unit 2007, Nash | Nexus Anchor Rain | 1111 Church Street, # 2007, Nashvil | Shared address tokens & high Jaro-Winkler |
| Helios | 66 Edgewood Street, Bridgeport, CT | @Helios | 66 Edgewood Street, Bridgeport, Con | Shared address tokens & high Jaro-Winkler |
| Helios | 66 Edgewood Street, Bridgeport, CT | Helios.Com | Bridgeport, 66 Edgewood Saint, Conn | Shared address tokens & high Jaro-Winkler |
| Unified Choice Dynamix | Unit BUILDING 3030, MD, 2701 Easter | Unified Choice Dynamx | MIDDLE RIVER, 2701 Eastern Blvd, MD | Shared address tokens & high Jaro-Winkler |
| Unified Choice Dynamix | Unit BUILDING 3030, MD, 2701 Easter | Unified Choice Center | 2701-A Eastern Boulevard, Unit Buil | Shared address tokens & high Jaro-Winkler |
