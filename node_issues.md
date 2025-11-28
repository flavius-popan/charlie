# Node Extraction Failures

- Special characters like ! after a name, "Good luck Charlie!". No Charlie.
- Too literal "90m nap" instead of "nap", pulls direct quotes, uses adjectives.
- Place is too narrow for home rooms ("living room", "closet", "balcony").
- Generic activities "preparing the room"
- No tenses, just direct quotes. Add V2 plan nuance.
- "Claudia" + "Claudia's", not sure how to fix...
- Single place based on name "Charlie's, Claudia's", "Jefe", "Jefe's"
- Dupliate "gym" + "going to the gym", "Uzbek restaurant" + "dinner at Uzbek restaurant"
- Conflict between action and person, "picking up Cody" vs just "Cody"
- How to handle upper/lower casing?
- Activities too direct from text "pairing on MIDILisp" vs "MIDILisp pairing"
- Low-value activities "responding to Discord messages"
- "using producer.ai", should verbs be captured in node name...? Probably not.
- "C/Zig/Zig Day/CtF/Soldering ", can't sub-extract, extra space (handle extra spaces in code?)
- "Aaktun in Durham", should be two locations
- "morning walk" + "evening walk", should be "walking/walks/walk"?
- Ensure "Suecia's mom" and "talked to mom" don't get confused
