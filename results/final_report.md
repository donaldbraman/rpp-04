# Geographic Policing Intensity Analysis - Final Report

## Executive Summary

This analysis implements the complete methodology guide for analyzing policing intensity patterns across 1076 census block groups. Using actual census population data, we identify significant disparities in arrest risks and enforcement patterns.

## Methodology Implementation

### Phase 1: Data Preparation and Geographic Categorization
- ✓ **Step 1**: Loaded 144,645 arrests (41,807 unique individuals)
- ✓ **Step 1A**: Obtained census data for 2980 block groups
- ✓ **Step 1B**: Merged census and arrest data (1076 matched block groups)
- ✓ **Step 2**: Identified discretionary arrests (56,138 / 144,645)
- ✓ **Step 3**: Calculated cut points using cumulative population distribution
- ✓ **Step 4**: Established three policing intensity categories

### Phase 2: Calculate Annual Arrest Risks
- ✓ **Step 5**: Calculated overall population annual risks
- ✓ **Step 6**: Calculated young men (18-35) annual risks
- ✓ **Step 7**: Projected lifetime risk probabilities

### Phase 3: Multiple Arrest and Escalation Analysis
- ✓ **Step 8**: Analyzed arrest frequency distribution
- ✓ **Steps 9-10**: Identified repeat offense patterns
- ✓ **Steps 11-12**: Calculated per capita escalation risks

### Phase 5: Drug Offense Deep Dive
- ✓ **Step 18**: Isolated 34,192 drug arrests
- ✓ **Steps 19-20**: Calculated drug arrest annual risks
- ✓ **Step 21**: Analyzed drug repeat patterns
- ✓ **Step 24**: Modeled enforcement under equal use assumption

## Key Findings

### Population Distribution (Census-Based)
- **Ultra-Policed**: 6.7% of population (134,522 people)
- **Highly Policed**: 15.4% of population (310,952 people)
- **Normally Policed**: 78.0% of population (1,574,964 people)

### Arrest Rates per 1,000 Population
| Category | Discretionary | Total | Unique Individuals |
|----------|--------------|-------|-------------------|
| Ultra-Policed | 160.0 | 379.7 | 100.8 |
| Highly Policed | 51.4 | 137.8 | 43.0 |
| Normally Policed | 4.5 | 11.4 | 4.1 |

### Annual Arrest Risk

**Overall Population:**
- Ultra-Policed: 0.96% (1 in 104)
- Normally Policed: 0.04% (1 in 2537)
- **Disparity: 24.4x**

**Young Men (18-35):**
- Ultra-Policed: 1.72% (1 in 58)
- Normally Policed: 0.08% (1 in 1231)
- **Disparity: 21.2x**

### Lifetime Risk (Young Men by Age 35)
- Ultra-Policed: 25.6%
- Normally Policed: 1.4%

### Drug Enforcement
- Total drug arrests: 34,192
- Unique individuals: 11,717
- Facing enhanced penalties (2+ arrests): 57.1%
- Facing mandatory minimums (3+ arrests): 36.5%

**Per Capita Drug Arrest Rates:**
- Ultra-Policed: 3.72 per 1,000 annually
- Normally Policed: 0.16 per 1,000 annually
- **Disparity: 22.9x**

### Arrest Frequency Patterns
- 34.1% have only 1 arrest
- 65.9% of those arrested are arrested again
- 72.1% of those with 2 arrests get a 3rd

## Disparities Summary

| Metric | Ultra-Policed | Normally Policed | Disparity Ratio |
|--------|--------------|------------------|-----------------|
| Overall Annual Risk | 0.96% | 0.04% | 24.4x |
| Young Men Annual Risk | 1.72% | 0.08% | 21.2x |
| Drug Arrests per 1,000 | 3.72 | 0.16 | 22.9x |

## Data Sources
- **Census Data**: 2980 block groups from ACS 5-year estimates
- **Total Population**: 4,936,378
- **Matched Population**: 2,020,438 (40.9% coverage)
- **Arrest Data**: 144,645 arrests over 10.5 years
- **Unique Individuals**: 41,807

## Methodology Notes
- Used actual census block group populations (not estimates)
- Based categories on discretionary arrests only
- Calculated risks using unique individuals (not total arrests)
- Applied methodology guide steps 1-24 systematically

---
*Analysis completed following methodology_guide.md*
*Generated: 2025-08-20 14:42:31*
