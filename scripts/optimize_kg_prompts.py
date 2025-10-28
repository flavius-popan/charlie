from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any, Iterable

import dspy
from dspy.teleprompt import BootstrapFewShot

from dspy_outlines import OutlinesAdapter, OutlinesLM
from dspy_outlines.kg_extraction import (
    Edge,
    KGExtractionModule,
    KnowledgeGraph,
    Node,
)

logger = logging.getLogger(__name__)

DEFAULT_OUTPUT = Path("prompts/kg_extraction_optimized.json")


def configure_lm(disable_prompt_cache: bool = True) -> None:
    """Configure DSPy with the Outlines-backed language model and adapter."""
    lm = OutlinesLM()
    adapter = OutlinesAdapter()
    dspy.configure(lm=lm, adapter=adapter)
    if disable_prompt_cache:
        logger.info(
            "Configured DSPy with OutlinesLM and OutlinesAdapter (prompt cache disabled)."
        )
    else:
        lm.enable_prompt_cache()
        logger.info(
            "Configured DSPy with OutlinesLM and OutlinesAdapter (prompt cache enabled)."
        )


def build_trainset() -> list[dspy.Example]:
    """Construct a diverse journal-style dataset for prompt optimization."""

    def make_example(text: str, nodes: list[Node], edges: list[Edge]) -> dspy.Example:
        guidance = (
            "\n\nReturn a JSON object with a top-level 'graph' field containing "
            "'nodes' and 'edges' arrays that capture the people, relationships, and feelings."
        )
        graph_obj = KnowledgeGraph(nodes=nodes, edges=edges)
        return dspy.Example(
            text=f"{text}{guidance}",
            graph=graph_obj.model_dump(),
        ).with_inputs("text")

    examples: list[dspy.Example] = [
        make_example(
            text=(
                "Dear diary, I'm fifteen and my hands shake thinking about tomorrow's "
                "science fair. Emma stayed late to glue the solar model, and Dad kept "
                "reminding me that my curiosity is bigger than the nerves."
            ),
            nodes=[
                Node(id=1, label="Author"),
                Node(id=2, label="Emma"),
                Node(id=3, label="Dad"),
                Node(id=4, label="Nervousness"),
                Node(id=5, label="Relief"),
            ],
            edges=[
                Edge(source=1, target=2, label="shares_nerves_with"),
                Edge(source=2, target=1, label="encourages"),
                Edge(source=1, target=3, label="feels_supported_by"),
                Edge(source=3, target=1, label="reassures"),
                Edge(source=1, target=4, label="feels_worried_about"),
                Edge(source=1, target=5, label="hopes_for"),
            ],
        ),
        make_example(
            text=(
                "Hey journal, Coach Maya pulled me aside after soccer drills and said "
                "my passes finally flow. Luis high-fived me so hard that I laughed, and "
                "I can feel confidence humming louder than the jitters."
            ),
            nodes=[
                Node(id=1, label="Author"),
                Node(id=2, label="Coach Maya"),
                Node(id=3, label="Luis"),
                Node(id=4, label="Confidence"),
                Node(id=5, label="Jitters"),
            ],
            edges=[
                Edge(source=2, target=1, label="inspires"),
                Edge(source=1, target=3, label="shares_excited_with"),
                Edge(source=3, target=1, label="boosts"),
                Edge(source=1, target=4, label="feels_growing"),
                Edge(source=1, target=5, label="lets_go_of"),
            ],
        ),
        make_example(
            text=(
                "I scribbled in the lunchroom today, and Ms. Tran said my poem about the "
                "tide sounded like a heartbeat. Kayla promised to read it at open mic, "
                "and suddenly the fear feels softer than her grin."
            ),
            nodes=[
                Node(id=1, label="Author"),
                Node(id=2, label="Ms. Tran"),
                Node(id=3, label="Kayla"),
                Node(id=4, label="Fear"),
                Node(id=5, label="Pride"),
            ],
            edges=[
                Edge(source=2, target=1, label="uplifts"),
                Edge(source=3, target=1, label="champions"),
                Edge(source=1, target=4, label="breathes_through"),
                Edge(source=1, target=5, label="lets_in"),
            ],
        ),
        make_example(
            text=(
                "Orientation day wiped me out, but Malik walked me around campus and "
                "told me the dining hall coffee is survivable. I called Mom and melted "
                "when she said she could hear excitement vibrating in my voice."
            ),
            nodes=[
                Node(id=1, label="Author"),
                Node(id=2, label="Malik"),
                Node(id=3, label="Mom"),
                Node(id=4, label="Excitement"),
                Node(id=5, label="Overwhelm"),
            ],
            edges=[
                Edge(source=2, target=1, label="grounds"),
                Edge(source=1, target=3, label="shares_buzz_with"),
                Edge(source=3, target=1, label="affirms"),
                Edge(source=1, target=4, label="feels_surging"),
                Edge(source=1, target=5, label="manages"),
            ],
        ),
        make_example(
            text=(
                "Career fair day: I clutched my portfolio while Priya squeezed my arm. "
                "When the recruiter smiled at my capstone story I nearly floated, and "
                "Priya's calm laugh kept my ego from spinning wild."
            ),
            nodes=[
                Node(id=1, label="Author"),
                Node(id=2, label="Priya"),
                Node(id=3, label="Recruiter"),
                Node(id=4, label="Ego"),
                Node(id=5, label="Calm"),
            ],
            edges=[
                Edge(source=2, target=1, label="steadies"),
                Edge(source=3, target=1, label="validates"),
                Edge(source=1, target=4, label="tempers"),
                Edge(source=1, target=5, label="leans_into"),
            ],
        ),
        make_example(
            text=(
                "First week at the design studio and I'm buzzing. Elena walked me through "
                "the client brief while jokes spilled everywhere, and Mateo texted to say "
                "my excitement feels electric, not reckless."
            ),
            nodes=[
                Node(id=1, label="Author"),
                Node(id=2, label="Elena"),
                Node(id=3, label="Mateo"),
                Node(id=4, label="Excitement"),
                Node(id=5, label="Reassurance"),
            ],
            edges=[
                Edge(source=2, target=1, label="guides"),
                Edge(source=1, target=3, label="shares_glow_with"),
                Edge(source=3, target=1, label="calms"),
                Edge(source=1, target=4, label="embraces"),
                Edge(source=1, target=5, label="soaks_up"),
            ],
        ),
        make_example(
            text=(
                "I told Jordan that dating in our twenties feels like a jazz duet—my "
                "heart riffed nervous while his patience wrapped around it. Later I sat "
                "with Grandma and let her stories slow my hopeful pulse."
            ),
            nodes=[
                Node(id=1, label="Author"),
                Node(id=2, label="Jordan"),
                Node(id=3, label="Grandma"),
                Node(id=4, label="Hope"),
                Node(id=5, label="Nerves"),
            ],
            edges=[
                Edge(source=1, target=2, label="feels_understood_by"),
                Edge(source=2, target=1, label="soothes"),
                Edge(source=3, target=1, label="calms"),
                Edge(source=1, target=4, label="holds"),
                Edge(source=1, target=5, label="softens"),
            ],
        ),
        make_example(
            text=(
                "Baby Mia screamed through sunrise but Theo brewed lavender tea and "
                "hugged me until the exhaustion unclenched. I texted Sasha because she "
                "always reminds me that tenderness counts as progress."
            ),
            nodes=[
                Node(id=1, label="Author"),
                Node(id=2, label="Theo"),
                Node(id=3, label="Sasha"),
                Node(id=4, label="Exhaustion"),
                Node(id=5, label="Tenderness"),
            ],
            edges=[
                Edge(source=2, target=1, label="supports"),
                Edge(source=1, target=3, label="seeks_comfort_from"),
                Edge(source=3, target=1, label="affirms"),
                Edge(source=1, target=4, label="releases"),
                Edge(source=1, target=5, label="cultivates"),
            ],
        ),
        make_example(
            text=(
                "Seminar prep almost swallowed me, so I met Lina at the library. We "
                "traded drafts, traded anxieties, and she reminded me to let the "
                "curiosity out louder than the imposter voice."
            ),
            nodes=[
                Node(id=1, label="Author"),
                Node(id=2, label="Lina"),
                Node(id=3, label="Curiosity"),
                Node(id=4, label="Imposter voice"),
            ],
            edges=[
                Edge(source=2, target=1, label="mirrors"),
                Edge(source=1, target=2, label="shares_fears_with"),
                Edge(source=1, target=3, label="amplifies"),
                Edge(source=1, target=4, label="shrinks"),
            ],
        ),
        make_example(
            text=(
                "Tonight's rehearsal glowed. Diego caught my eye when my solo wobbled, "
                "and his smile steadied the tremor. I journaled under the stage lights, "
                "letting gratitude hum through my ribs."
            ),
            nodes=[
                Node(id=1, label="Author"),
                Node(id=2, label="Diego"),
                Node(id=3, label="Gratitude"),
                Node(id=4, label="Tremor"),
            ],
            edges=[
                Edge(source=2, target=1, label="steadies"),
                Edge(source=1, target=2, label="trusts"),
                Edge(source=1, target=3, label="bathes_in"),
                Edge(source=1, target=4, label="calms_down"),
            ],
        ),
        make_example(
            text=(
                "Marathon training at thirty-three feels like negotiating with my knees. "
                "Sharon biked beside me repeating that the finish line lives in my grin, "
                "and Ravi sent a voice memo praising my stubborn joy."
            ),
            nodes=[
                Node(id=1, label="Author"),
                Node(id=2, label="Sharon"),
                Node(id=3, label="Ravi"),
                Node(id=4, label="Joy"),
                Node(id=5, label="Doubt"),
            ],
            edges=[
                Edge(source=2, target=1, label="motivates"),
                Edge(source=3, target=1, label="cheers"),
                Edge(source=1, target=4, label="fuels"),
                Edge(source=1, target=5, label="pushes_past"),
            ],
        ),
        make_example(
            text=(
                "Dad called from physical therapy and I could hear the tired pride in "
                "his voice. I promised to visit Friday with soup, and Mara wrote me a "
                "note saying caretaking still counts as dreaming."
            ),
            nodes=[
                Node(id=1, label="Author"),
                Node(id=2, label="Dad"),
                Node(id=3, label="Mara"),
                Node(id=4, label="Pride"),
                Node(id=5, label="Tiredness"),
            ],
            edges=[
                Edge(source=2, target=1, label="shows_trust"),
                Edge(source=1, target=2, label="feels_devoted_to"),
                Edge(source=3, target=1, label="encourages"),
                Edge(source=1, target=4, label="shares"),
                Edge(source=1, target=5, label="acknowledges"),
            ],
        ),
        make_example(
            text=(
                "I resigned today. Leo stayed on the call until I stopped shaking, and "
                "my sister Mia sent voice notes reminding me that reinvention can taste "
                "like relief instead of failure."
            ),
            nodes=[
                Node(id=1, label="Author"),
                Node(id=2, label="Leo"),
                Node(id=3, label="Mia"),
                Node(id=4, label="Relief"),
                Node(id=5, label="Failure"),
            ],
            edges=[
                Edge(source=2, target=1, label="steadying"),
                Edge(source=3, target=1, label="reframes"),
                Edge(source=1, target=4, label="welcomes"),
                Edge(source=1, target=5, label="releases"),
            ],
        ),
        make_example(
            text=(
                "The twins left for college and the house echoes. I brewed cinnamon tea "
                "while Carlos squeezed my shoulder, and I let the quiet grief sit beside "
                "his warm optimism."
            ),
            nodes=[
                Node(id=1, label="Author"),
                Node(id=2, label="Carlos"),
                Node(id=3, label="Grief"),
                Node(id=4, label="Optimism"),
            ],
            edges=[
                Edge(source=2, target=1, label="comforts"),
                Edge(source=1, target=3, label="sits_with"),
                Edge(source=1, target=4, label="leans_toward"),
            ],
        ),
        make_example(
            text=(
                "My granddaughter Iris FaceTimed me to show her art project, and the "
                "colors woke something gentle. Harold listened as I confessed how lonely "
                "the apartment feels at dusk, then he invited me to choir practice."
            ),
            nodes=[
                Node(id=1, label="Author"),
                Node(id=2, label="Iris"),
                Node(id=3, label="Harold"),
                Node(id=4, label="Gentleness"),
                Node(id=5, label="Loneliness"),
            ],
            edges=[
                Edge(source=2, target=1, label="brightens"),
                Edge(source=3, target=1, label="invites"),
                Edge(source=1, target=4, label="awakens"),
                Edge(source=1, target=5, label="acknowledges"),
            ],
        ),
        make_example(
            text=(
                "Volunteering at the community garden keeps my seventy-year-old knees "
                "honest. Tasha hands me mint tea and tells me the kids feel safe with me, "
                "and I let that pride stretch taller than the aches."
            ),
            nodes=[
                Node(id=1, label="Author"),
                Node(id=2, label="Tasha"),
                Node(id=3, label="Kids"),
                Node(id=4, label="Pride"),
                Node(id=5, label="Aches"),
            ],
            edges=[
                Edge(source=2, target=1, label="thanks"),
                Edge(source=3, target=1, label="trusts"),
                Edge(source=1, target=4, label="feels_lifted"),
                Edge(source=1, target=5, label="accepts"),
            ],
        ),
        make_example(
            text=(
                "Doctor Patel adjusted my meds today, and my grandson Noah held my hand "
                "through the waiting room buzz. I told him the fear shrinks when he hums "
                "our silly song, and he grinned wider than the hospital lights."
            ),
            nodes=[
                Node(id=1, label="Author"),
                Node(id=2, label="Doctor Patel"),
                Node(id=3, label="Noah"),
                Node(id=4, label="Fear"),
                Node(id=5, label="Comfort"),
            ],
            edges=[
                Edge(source=2, target=1, label="guides"),
                Edge(source=3, target=1, label="soothes"),
                Edge(source=1, target=4, label="shrinks"),
                Edge(source=1, target=5, label="absorbs"),
            ],
        ),
        make_example(
            text=(
                "Thirteen-year-old me survived the debate meet! Jamal bumped my shoulder "
                "and said my rebuttal sounded fierce, and Mom tucked a victory donut into "
                "my hand while my adrenaline simmered."
            ),
            nodes=[
                Node(id=1, label="Author"),
                Node(id=2, label="Jamal"),
                Node(id=3, label="Mom"),
                Node(id=4, label="Adrenaline"),
                Node(id=5, label="Pride"),
            ],
            edges=[
                Edge(source=2, target=1, label="celebrates"),
                Edge(source=3, target=1, label="nurtures"),
                Edge(source=1, target=4, label="settles"),
                Edge(source=1, target=5, label="glows"),
            ],
        ),
        make_example(
            text=(
                "Study group tonight felt heroic. Mei organized flashcards while Rowan "
                "kept refilling tea, and I let my stress untangle as their patience "
                "wrapped around me."
            ),
            nodes=[
                Node(id=1, label="Author"),
                Node(id=2, label="Mei"),
                Node(id=3, label="Rowan"),
                Node(id=4, label="Stress"),
                Node(id=5, label="Patience"),
            ],
            edges=[
                Edge(source=2, target=1, label="structures"),
                Edge(source=3, target=1, label="supports"),
                Edge(source=1, target=4, label="releases"),
                Edge(source=1, target=5, label="absorbs"),
            ],
        ),
        make_example(
            text=(
                "As a sophomore RA, I talked Avery through a homesick spiral. After the "
                "door closed I called my brother Omar and let him hear how proud and "
                "tender I felt."
            ),
            nodes=[
                Node(id=1, label="Author"),
                Node(id=2, label="Avery"),
                Node(id=3, label="Omar"),
                Node(id=4, label="Tenderness"),
                Node(id=5, label="Homesickness"),
            ],
            edges=[
                Edge(source=1, target=2, label="comforts"),
                Edge(source=2, target=1, label="trusts"),
                Edge(source=1, target=3, label="shares_pride_with"),
                Edge(source=1, target=4, label="embraces"),
                Edge(source=1, target=5, label="relates_to"),
            ],
        ),
        make_example(
            text=(
                "My mentor Zahra reviewed my portfolio and circled the pieces that thrum "
                "with softness. I emailed Kian the updates and admitted I'm equal parts "
                "thrilled and terrified to show the gallery."
            ),
            nodes=[
                Node(id=1, label="Author"),
                Node(id=2, label="Zahra"),
                Node(id=3, label="Kian"),
                Node(id=4, label="Thrill"),
                Node(id=5, label="Terror"),
            ],
            edges=[
                Edge(source=2, target=1, label="guides"),
                Edge(source=1, target=3, label="confides_in"),
                Edge(source=1, target=4, label="embraces"),
                Edge(source=1, target=5, label="manages"),
            ],
        ),
        make_example(
            text=(
                "Rainy Sunday with Dad playing chess. He beat me twice but kept pointing "
                "out the hopeful moves I attempted, and I let the quiet joy hum louder "
                "than the losses."
            ),
            nodes=[
                Node(id=1, label="Author"),
                Node(id=2, label="Dad"),
                Node(id=3, label="Joy"),
                Node(id=4, label="Frustration"),
            ],
            edges=[
                Edge(source=2, target=1, label="teaches"),
                Edge(source=1, target=2, label="appreciates"),
                Edge(source=1, target=3, label="tunes_into"),
                Edge(source=1, target=4, label="lets_go_of"),
            ],
        ),
        make_example(
            text=(
                "Community college night class left me buzzing. Professor Lane praised my "
                "essay for sounding like lived truth, and Nina squeezed my hand whispering "
                "that my courage is contagious."
            ),
            nodes=[
                Node(id=1, label="Author"),
                Node(id=2, label="Professor Lane"),
                Node(id=3, label="Nina"),
                Node(id=4, label="Courage"),
                Node(id=5, label="Buzz"),
            ],
            edges=[
                Edge(source=2, target=1, label="validates"),
                Edge(source=3, target=1, label="celebrates"),
                Edge(source=1, target=4, label="expands"),
                Edge(source=1, target=5, label="rides"),
            ],
        ),
        make_example(
            text=(
                "At thirty-nine I joined a ceramics class. Instructor Paul laughed when "
                "my bowl wobbled and showed me how patience feels in the wrists. Later, "
                "Ari texted me a playlist that matched my mellow pride."
            ),
            nodes=[
                Node(id=1, label="Author"),
                Node(id=2, label="Paul"),
                Node(id=3, label="Ari"),
                Node(id=4, label="Pride"),
                Node(id=5, label="Patience"),
            ],
            edges=[
                Edge(source=2, target=1, label="encourages"),
                Edge(source=3, target=1, label="celebrates"),
                Edge(source=1, target=4, label="feels"),
                Edge(source=1, target=5, label="practices"),
            ],
        ),
        make_example(
            text=(
                "I met with Pastor Elena to talk about grieving Mom. She held space while "
                "I cried, then invited me to light a candle. Afterward, Malik walked me "
                "home and let his steady kindness quiet the ache."
            ),
            nodes=[
                Node(id=1, label="Author"),
                Node(id=2, label="Pastor Elena"),
                Node(id=3, label="Malik"),
                Node(id=4, label="Grief"),
                Node(id=5, label="Kindness"),
            ],
            edges=[
                Edge(source=2, target=1, label="comforts"),
                Edge(source=3, target=1, label="steadies"),
                Edge(source=1, target=4, label="honors"),
                Edge(source=1, target=5, label="absorbs"),
            ],
        ),
        make_example(
            text=(
                "Grandpa journal: the morning chess club crowned me puzzle king. "
                "Margaret teased that my grin looked twenty again, and I felt my "
                "confidence stretch taller than the aches in my hands."
            ),
            nodes=[
                Node(id=1, label="Author"),
                Node(id=2, label="Margaret"),
                Node(id=3, label="Chess club"),
                Node(id=4, label="Confidence"),
                Node(id=5, label="Aches"),
            ],
            edges=[
                Edge(source=3, target=1, label="celebrates"),
                Edge(source=2, target=1, label="teases"),
                Edge(source=1, target=4, label="swells"),
                Edge(source=1, target=5, label="outpaces"),
            ],
        ),
        make_example(
            text=(
                "Eighth-grade Bella wrote a song about the school dance. My brother Eli "
                "helped with chords and kept laughing whenever my voice cracked, and I "
                "felt goofy joy chase away every anxious flutter."
            ),
            nodes=[
                Node(id=1, label="Author"),
                Node(id=2, label="Eli"),
                Node(id=3, label="Goofy joy"),
                Node(id=4, label="Anxious flutter"),
            ],
            edges=[
                Edge(source=2, target=1, label="supports"),
                Edge(source=1, target=2, label="relies_on"),
                Edge(source=1, target=3, label="embraces"),
                Edge(source=1, target=4, label="chases"),
            ],
        ),
        make_example(
            text=(
                "Graduate fellowship update: Dr. Singh said my proposal pulses with care. "
                "I texted Bea to scream about it, and she replied with voice notes that "
                "made my gratitude ring clear."
            ),
            nodes=[
                Node(id=1, label="Author"),
                Node(id=2, label="Dr. Singh"),
                Node(id=3, label="Bea"),
                Node(id=4, label="Gratitude"),
                Node(id=5, label="Care"),
            ],
            edges=[
                Edge(source=2, target=1, label="affirms"),
                Edge(source=1, target=3, label="celebrates_with"),
                Edge(source=3, target=1, label="echoes"),
                Edge(source=1, target=4, label="rings"),
                Edge(source=1, target=5, label="pours_into"),
            ],
        ),
        make_example(
            text=(
                "Dear sketchbook, fourteen feels like balancing on a beam. Nia texted me "
                "a silly meme until my panic softened, and Coach Reed reminded me my "
                "focus is braver than the wobble."
            ),
            nodes=[
                Node(id=1, label="Author"),
                Node(id=2, label="Nia"),
                Node(id=3, label="Coach Reed"),
                Node(id=4, label="Panic"),
                Node(id=5, label="Focus"),
            ],
            edges=[
                Edge(source=2, target=1, label="lightens"),
                Edge(source=3, target=1, label="encourages"),
                Edge(source=1, target=4, label="softens"),
                Edge(source=1, target=5, label="trusts"),
            ],
        ),
        make_example(
            text=(
                "Parenting teens is wild. Maya confessed she feels invisible sometimes, "
                "so I held her hand and told her the truth glows in her sketches. Later, "
                "Ken and I traded relief in the lamplight."
            ),
            nodes=[
                Node(id=1, label="Author"),
                Node(id=2, label="Maya"),
                Node(id=3, label="Ken"),
                Node(id=4, label="Relief"),
                Node(id=5, label="Invisible feeling"),
            ],
            edges=[
                Edge(source=1, target=2, label="reassures"),
                Edge(source=2, target=1, label="trusts"),
                Edge(source=1, target=3, label="shares_relief_with"),
                Edge(source=1, target=4, label="breathes"),
                Edge(source=1, target=5, label="lightens"),
            ],
        ),
        make_example(
            text=(
                "Neighborhood council meeting left my heart buzzing. Omar thanked me for "
                "speaking up about the playground lights, and Li slipped me a note saying "
                "my steady hope keeps her organizing."
            ),
            nodes=[
                Node(id=1, label="Author"),
                Node(id=2, label="Omar"),
                Node(id=3, label="Li"),
                Node(id=4, label="Hope"),
                Node(id=5, label="Determination"),
            ],
            edges=[
                Edge(source=2, target=1, label="appreciates"),
                Edge(source=3, target=1, label="encourages"),
                Edge(source=1, target=4, label="sustains"),
                Edge(source=1, target=5, label="sparks"),
            ],
        ),
        make_example(
            text=(
                "We hosted the neighborhood book club. Janice finally spoke about her "
                "divorce, and I watched Alana slide tissues across the table. I felt my "
                "own empathy swell like the tea kettle."
            ),
            nodes=[
                Node(id=1, label="Author"),
                Node(id=2, label="Janice"),
                Node(id=3, label="Alana"),
                Node(id=4, label="Empathy"),
                Node(id=5, label="Comfort"),
            ],
            edges=[
                Edge(source=2, target=3, label="accepts"),
                Edge(source=3, target=2, label="comforts"),
                Edge(source=1, target=4, label="swells"),
                Edge(source=1, target=5, label="offers"),
            ],
        ),
        make_example(
            text=(
                "Letter to myself at sixty: Ava stopped by with fresh sourdough and we "
                "laughed about our grandkids' slang. After she left, I let the gentle joy "
                "and the quiet ache sit together on the couch."
            ),
            nodes=[
                Node(id=1, label="Author"),
                Node(id=2, label="Ava"),
                Node(id=3, label="Gentle joy"),
                Node(id=4, label="Quiet ache"),
            ],
            edges=[
                Edge(source=2, target=1, label="uplifts"),
                Edge(source=1, target=2, label="cherishes"),
                Edge(source=1, target=3, label="savors"),
                Edge(source=1, target=4, label="acknowledges"),
            ],
        ),
        make_example(
            text=(
                "Morning pages at fifty-five: I wrote about hiking with Lena. She matched "
                "my pace and kept praising my stubborn spirit, and I felt the gratitude "
                "settle into every muscle."
            ),
            nodes=[
                Node(id=1, label="Author"),
                Node(id=2, label="Lena"),
                Node(id=3, label="Stubborn spirit"),
                Node(id=4, label="Gratitude"),
            ],
            edges=[
                Edge(source=2, target=1, label="encourages"),
                Edge(source=1, target=2, label="appreciates"),
                Edge(source=2, target=3, label="celebrates"),
                Edge(source=1, target=4, label="absorbs"),
            ],
        ),
        make_example(
            text=(
                "Assisted living journal: Nurse Priya sat with me through the thunderstorm "
                "and let me squeeze her hand. My son Caleb called afterward, and between "
                "them my fear eased into sleepy calm."
            ),
            nodes=[
                Node(id=1, label="Author"),
                Node(id=2, label="Nurse Priya"),
                Node(id=3, label="Caleb"),
                Node(id=4, label="Fear"),
                Node(id=5, label="Calm"),
            ],
            edges=[
                Edge(source=2, target=1, label="comforts"),
                Edge(source=3, target=1, label="reassures"),
                Edge(source=1, target=4, label="softens"),
                Edge(source=1, target=5, label="settles_into"),
            ],
        ),
        make_example(
            text=(
                "Youth group tonight left my heart loud. Tori cried about exams, and I "
                "showed her the breathing trick Aunt Lila taught me. Later Aunt Lila "
                "texted to say she's proud I'm passing the calm along."
            ),
            nodes=[
                Node(id=1, label="Author"),
                Node(id=2, label="Tori"),
                Node(id=3, label="Aunt Lila"),
                Node(id=4, label="Calm"),
                Node(id=5, label="Stress"),
            ],
            edges=[
                Edge(source=1, target=2, label="soothes"),
                Edge(source=2, target=1, label="trusts"),
                Edge(source=3, target=1, label="praises"),
                Edge(source=1, target=4, label="shares"),
                Edge(source=1, target=5, label="dissolves"),
            ],
        ),
        make_example(
            text=(
                "Senior center newsletter: I led the storytelling circle today. When Ruth "
                "described her wartime letters, everyone leaned in, and I felt the room's "
                "collective tenderness cradle her memories."
            ),
            nodes=[
                Node(id=1, label="Author"),
                Node(id=2, label="Ruth"),
                Node(id=3, label="Storytelling circle"),
                Node(id=4, label="Tenderness"),
            ],
            edges=[
                Edge(source=1, target=3, label="guides"),
                Edge(source=2, target=3, label="shares"),
                Edge(source=1, target=4, label="feels_rising"),
                Edge(source=3, target=2, label="supports"),
            ],
        ),
        make_example(
            text=(
                "Freshman dorm whisper: I told Casey I'm scared the roommate will hate my "
                "playlist. Casey laughed and said my joy is the reason she picked me, and "
                "suddenly my chest felt bright."
            ),
            nodes=[
                Node(id=1, label="Author"),
                Node(id=2, label="Casey"),
                Node(id=3, label="Joy"),
                Node(id=4, label="Fear"),
            ],
            edges=[
                Edge(source=2, target=1, label="reassures"),
                Edge(source=1, target=2, label="confides_in"),
                Edge(source=1, target=3, label="brightens"),
                Edge(source=1, target=4, label="releases"),
            ],
        ),
        make_example(
            text=(
                "Diary of a sixty-eight-year-old widow: Sam visited with tulips and we "
                "sat on the porch naming the shades. I let the gratitude bloom until the "
                "ache loosened its grip."
            ),
            nodes=[
                Node(id=1, label="Author"),
                Node(id=2, label="Sam"),
                Node(id=3, label="Gratitude"),
                Node(id=4, label="Ache"),
            ],
            edges=[
                Edge(source=2, target=1, label="comforts"),
                Edge(source=1, target=2, label="cherishes"),
                Edge(source=1, target=3, label="blooms"),
                Edge(source=1, target=4, label="loosens"),
            ],
        ),
    ]

    logger.info("Prepared %d labeled examples for optimization.", len(examples))
    return examples


def _coerce_graph(graph_like: Any) -> KnowledgeGraph | None:
    """Convert dictionaries or Pydantic objects into a KnowledgeGraph."""
    if graph_like is None:
        return None
    if isinstance(graph_like, KnowledgeGraph):
        return graph_like
    if isinstance(graph_like, dict):
        try:
            return KnowledgeGraph.model_validate(graph_like)
        except Exception as exc:  # noqa: BLE001
            logger.debug("Failed to coerce graph dict: %s", exc)
    return None


def kg_quality_metric(example: dspy.Example, prediction, trace=None) -> float:
    """Simple quality metric that checks structural validity and label coverage."""
    predicted_graph = getattr(prediction, "graph", None)
    predicted_graph = _coerce_graph(predicted_graph)
    if predicted_graph is None:
        return 0.0

    graph_dict = predicted_graph.model_dump()
    nodes = graph_dict.get("nodes") or []
    edges = graph_dict.get("edges") or []
    if not nodes or not edges:
        return 0.0

    expected_graph = _coerce_graph(getattr(example, "graph", None))
    if expected_graph is None:
        return 0.0

    expected_labels = {node.label for node in expected_graph.nodes}
    predicted_labels = {node["label"] for node in nodes}
    if not expected_labels:
        return 1.0

    coverage = len(expected_labels & predicted_labels) / len(expected_labels)
    return coverage


class ObservableTrainset(list):
    """List wrapper that logs progress as the optimizer consumes examples."""

    def __iter__(self):
        total = len(self)
        for idx, example in enumerate(super().__iter__(), start=1):
            logger.info("Optimizer consuming example %d of %d", idx, total)
            yield example


def evaluate_module(
    module: KGExtractionModule, dataset: Iterable[dspy.Example]
) -> float:
    """Compute average metric score for the given module on the dataset."""
    scores: list[float] = []
    for example in dataset:
        prediction = module(text=example.text, known_entities=None)
        score = kg_quality_metric(example, prediction)
        scores.append(score)
    return sum(scores) / len(scores) if scores else 0.0


def optimize_prompts(
    trainset: list[dspy.Example],
    max_bootstrapped_demos: int,
    max_labeled_demos: int,
) -> KGExtractionModule:
    """Run BootstrapFewShot to optimize the knowledge graph prompt."""
    try:
        total = len(trainset)
    except TypeError:
        total = "unknown"
    logger.info("Starting BootstrapFewShot optimization over %s examples.", total)
    optimizer = BootstrapFewShot(
        metric=kg_quality_metric,
        max_bootstrapped_demos=max_bootstrapped_demos,
        max_labeled_demos=max_labeled_demos,
    )
    student = KGExtractionModule()
    optimized = optimizer.compile(student=student, trainset=trainset)
    logger.info(
        "Optimization completed with %d bootstrapped demos.", max_bootstrapped_demos
    )
    return optimized


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Optimize knowledge-graph prompts using DSPy BootstrapFewShot."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Path to save optimized prompts (default: {DEFAULT_OUTPUT}).",
    )
    parser.add_argument(
        "--max-bootstrapped-demos",
        type=int,
        default=5,
        help="Maximum number of bootstrapped demos generated by the optimizer.",
    )
    parser.add_argument(
        "--max-labeled-demos",
        type=int,
        default=3,
        help="Maximum number of labeled demos drawn from the trainset.",
    )
    parser.add_argument(
        "--no-eval",
        action="store_true",
        help="Skip before/after evaluation against the training set.",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    args = parse_args()

    configure_lm()
    trainset = build_trainset()

    if not args.no_eval:
        baseline_student = KGExtractionModule()
        baseline_score = evaluate_module(baseline_student, trainset)
        logger.info("Baseline average metric: %.3f", baseline_score)

    optimized_module = optimize_prompts(
        trainset=ObservableTrainset(trainset),
        max_bootstrapped_demos=args.max_bootstrapped_demos,
        max_labeled_demos=args.max_labeled_demos,
    )

    if not args.no_eval:
        optimized_score = evaluate_module(optimized_module, trainset)
        logger.info("Optimized average metric: %.3f", optimized_score)

    output_path = args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    optimized_module.save_prompts(str(output_path))
    logger.info("Saved optimized prompts to %s", output_path)


if __name__ == "__main__":
    main()
