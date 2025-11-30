# Node Extraction Failures

NOTE: When adding these, validate that a given prompt currently reproduces the issues listed,
then ensure the positive examples show the intended results!

---

## Guiding Principle: Entities Compose with Edges

**Entity names should be noun phrases that compose naturally with edges.**

The V2 edge model uses verbs like `Met`, `Visited`, `SpendsTimeWith`, `ParticipatesIn` to capture
relationships and actions. The edge carries the semantic relationship, so the entity should be
the pure noun form.

| Text in Journal | Entity (noun) | Edge (verb) |
|-----------------|---------------|-------------|
| "went to the gym" | gym | Visited |
| "morning walk with Sarah" | walking, Sarah | ParticipatesIn, SpendsTimeWith |
| "picking up Cody" | Cody | (action implicit in episode) |
| "dinner at Uzbek restaurant" | Uzbek restaurant | Visited |

**Test**: Does "I [EdgeVerb] [EntityName]" read naturally?
- "I ParticipatesIn walking" - yes
- "I ParticipatesIn morning walk" - awkward
- "I SpendsTimeWith Cody" - yes
- "I SpendsTimeWith picking up Cody" - no

---

## Category 1: Missing Entities (Recognition Gaps)

Entities present in text but not extracted.

- Misses sequential names, "Jerry, Bunicu, ..." or just doesn't get all names.
- Special characters like ! after a name, "Good luck Charlie!". No Charlie.

**Notes for examples:**
- Need an example with 3+ names in a list/sequence
- Need an example with names adjacent to punctuation (!, ?, quotes)
- Consider names at sentence boundaries or in exclamations

---

## Category 2: Over-Literal Extraction (Verbatim Copying)

Extracts text too literally instead of normalizing to canonical entity form.

- Too literal "90m nap" instead of "nap", pulls direct quotes, uses adjectives.
- Activities too direct from text "pairing on MIDILisp" vs "MIDILisp pairing"
- "using producer.ai" vs "producer.ai"

**Notes for examples:**
- Entity names should be **noun phrases**, not verb phrases
- Aligns with V2 edge model: edges carry the action/verb (ParticipatesIn, SpendsTimeWith)
- "I was running at the gym" -> entity: "gym", "running" (not "running at the gym")
- Modifiers like duration (90m), time-of-day (morning), adjectives should be stripped
- Examples should show: raw text with modifiers -> clean entity name

---

## Category 3: Normalization/Deduplication (Same Entity, Different Forms)

Same conceptual entity extracted multiple times in different surface forms.

- "Claudia" + "Claudia's"
- Duplicate "gym" + "going to the gym", "Uzbek restaurant" + "dinner at Uzbek restaurant"
- "morning walk" + "evening walk", should be "walking/walks/walk"?
- How to handle upper/lower casing?

**Notes for examples:**
- Possessives should normalize: "Claudia's house" -> entities: "Claudia", NOT "Claudia's"
- Activities should be base form for edge composition: "ParticipatesIn walking" (not "morning walk")
- Venue + activity combos: extract venue AND activity separately, not combined
- Examples should demonstrate picking ONE canonical form when multiple variants appear

---

## Category 4: Granularity Issues (Too Narrow or Too Generic)

Entities at wrong level of specificity.

**Too narrow:**
- Place is too narrow for home rooms ("living room", "closet", "balcony").

**Too generic/low-value:**
- Generic activities "preparing the room"
- Low-value activities "responding to Discord messages"

**Notes for examples:**
- Home rooms aren't places (you don't "VisitedPlace living room")
- But named venues within a place ARE entities: "the kitchen at Mom's house" -> "Mom's house"
- Activities should be recurring/meaningful, not one-off chores
- Test: would you want to query "show me all times I [activity]"? If no, don't extract.

---

## Category 5: Compound Entity Parsing (Multiple Entities in One Phrase)

Phrases containing multiple entities that should be separated.

- "C/Zig/Zig Day/CtF/Soldering ", can't sub-extract, extra space (handle extra spaces in code?)
- "Aaktun in Durham", should be two locations
- Single place based on name "Charlie's, Claudia's", "Jefe", "Jefe's"

**Notes for examples:**
- Slash-separated lists should produce multiple entities
- "X in Y" for locations: X is venue/restaurant, Y is city -> both are Place entities
- Possessive place names: "Charlie's" (restaurant) is a Place, distinct from "Charlie" (Person)
- Need examples showing: compound phrase -> list of separate entities

---

## Category 6: Entity Type Confusion (Action vs Entity)

Confusing an action/event with the entity involved.

- Conflict between action and person, "picking up Cody" vs just "Cody"
- Ensure "Suecia's mom" and "talked to mom" don't get confused

**Notes for examples:**
- Actions belong in EDGES, not entity names
- "picking up Cody" -> entity: "Cody" (Person), edge will capture the action
- Relational references: "Suecia's mom" is a Person (relationship to Suecia)
- "talked to mom" -> "mom" is Person, "talking" could be Activity if recurring
- The V2 edge model handles verbs: Met, Visited, SpendsTimeWith, etc.

---

## Example Coverage Strategy

Aim for 3-7 new examples that hit multiple categories each:

| Example Focus | Categories Covered |
|--------------|-------------------|
| List of names with punctuation | 1, 6 |
| Activity with modifiers + venue | 2, 3, 4 |
| Compound location phrase | 5, 3 |
| Possessives and relationships | 3, 5, 6 |
| Low-value vs meaningful activities | 2, 4 |

Each example should show the WRONG extraction (what current prompt produces) implicitly
by having expected_entities that demonstrate the CORRECT behavior.
