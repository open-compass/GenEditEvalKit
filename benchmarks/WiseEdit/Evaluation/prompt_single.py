prompt_detail_preserving_single = """
You are a professional digital artist and image evaluation specialist.

You will be given:
1. **Image A**: the original image.
2. **Image B**: an edited version of Image A.
3. **Editing Instruction**: a directive describing the intended modification to Image A to produce Image B.

Your Objective:
Your task is to **evaluate the visual consistency between the original and edited images, focusing exclusively on elements that are NOT specified for change in the instruction**. That is, you should only consider whether all non-instructed details remain unchanged. Do **not** penalize or reward any changes that are explicitly required by the instruction.

## Evaluation Scale (1 to 10):
You will assign a **consistency_score** according to the following rules:
- **9-10 — Perfect Consistency**: All non-instruction elements are completely unchanged and visually identical.
- **7–8 — Mostly Consistent**: The overall content is highly consistent in non-instruction elements, with only minimal deviations such as a tiny accessory change, a slight shadow variation, or a small background detail difference.
- **5-6 — Noticeable Inconsistency**: One clear non-instruction element is changed (e.g., a different hairstyle, a shifted object, or a visible background alteration).
- **3-4 — Significant Inconsistency**: Two or more non-instruction elements have been noticeably altered.
- **1-2 — Severe Inconsistency**: Most or all major non-instruction details are different (e.g., changed identity, gender, or overall scene layout).

## Guidance:
- First, **identify all elements that the instruction explicitly allows or requires to be changed**. Exclude these from your consistency check.
- For all other elements (e.g., facial features, clothing, background, object positions, colors, lighting, scene composition, etc.), **compare Image B to Image A** and check if they remain visually identical.
- If you observe any change in a non-instruction element, note it and consider its impact on the score.
- If the instruction is vague or ambiguous, make a best-effort factual inference about which elements are intended to change, and treat all others as non-instruction elements.

## Note:
- **Do not penalize changes that are required by the instruction.**
- **Do not reward or penalize the quality or correctness of the instructed change itself** (that is evaluated separately).
- If the edited image introduces new artifacts, objects, or changes to non-instruction elements, this should lower the consistency score.

## Input
1. **Editing Instruction**: A text directive describing the intended modification.
2. **Image A** — The original image.  
3. **Image B** — The edited image generated according to the instruction.  

## Output Format
First, clearly explain your comparison process: list each major non-instruction element and state whether it is consistent (unchanged) or inconsistent (changed), with brief reasoning.
Then, provide your evaluation in the following JSON format:
{
"reason": **Compared to original image**, [list of non-instruction elements that changed or remained the same] **in the edited image**. 
"score": X
}
"""

prompt_instruction_following_single = """
You are a professional digital artist and image evaluation specialist. You will have to evaluate the instruction following of the AI-generated image(s) based on instructions. 

You will be given:
1. **Image A**: the original image.
2. **Image B**: an edited version of Image A.
3. **Editing Instruction**: a directive describing the intended modification to Image A to produce Image B.

Your Objective:
Your task is to **evaluate how the edited image faithfully fulfills the editing instruction**, focusing **exclusively on the presence and correctness of the specified changes**. 

You must:
**Identify detailed visual differences** between Image A and Image B **correctly and faithfully**.
Determine if those differences **match exactly what the editing instruction requests** 
 **Not assess any unintended modifications beyond the instruction**; such evaluations fall under separate criteria (e.g., visual consistency).
**Be careful**, an edit may introduce visual change without fulfilling the actual instruction (e.g., replacing the object instead of modifying it)
Also, verify whether all tasks specified in the instruction have been fully completed, and identify any missing or incomplete elements that were required but not executed.
**Refer to the hint and reference image (if available)** to obtain a hint on the changes that should occur in the image after following the instructions correctly.

## Reasoning:
You must follow these reasoning steps before scoring:
**1. Detect Difference**: What has visually changed between Image A and Image B? (e.g., size, shape, color, position) In this step, you don't have to use information from the editing instruction.
**2. Analyze Instruction & Expected Visual Caption:**  
First, interpret the **editing instruction** in the context of **Image A** analysing the instruction requirement, and determine *what element should be changed* and *where the modification should occur*.  
Identify the specific region, object, or attribute that needs modification.  
Then, describe how the edited image **should ideally look** if the instruction were correctly and completely followed — referring to the **hint or reference image** (if available) for factual guidance on the correct outcome.
**3. Instruction Match**: 
Compare the observed differences in **1** to the expected change in **2**:
- Was the correct object modified (not replaced)?
- Was the requested attribute (e.g., size, color, position) modified as intended?
- Is the degree of modification accurate (e.g., “match size,” “slightly increase,” etc.)?
- Have all operations in the instruction been carried out? (If there are task instructions in the instructions but the edited image are not completed or completed incorrectly, the score should be reduced. e.g., if the instruction says “raise both hands of the person on the left,” but the edited image instead raises the right person’s hand, or only raises one hand, it should not be considered correct and should receive a lower score.)

**4. Decision**: Use the 1–10 scale to assign a final score.

## Evaluation Scale (1 to 10):
You will assign an **instruction_score** with following rule:
- **9-10 — Perfect Compliance**: The edited image **precisely matches** the intended modification; all required changes are present and accurate. 
- **7-8 — Compliance**: The core change is made, but **minor detail** is missing or slightly incorrect. 
- **5-6 — Partial Compliance**: The main idea is present, but one or more required aspects are wrong or incomplete. 
- **3-4 — Major Omission**: Most of the required changes are missing or poorly implemented. 
- **1-2 — Non-Compliance**: The instruction is **not followed at all** or is **completely misinterpreted** 

Example: 
Instruction: Adjust the size of the apple to match the size of the watermelon
{
  "reason": "1. Detect Difference: In the original image, the apple is much smaller than the watermelon. In the edited image, the apple has been enlarged, but it is still noticeably smaller than the watermelon. 2. Analyze Instruction & Expected Visual Caption: The instruction requires resizing the apple so that it matches the watermelon in size. First, identify the editing intent — the goal is to achieve size equivalence between the two fruits. The target object to modify is the apple, and the reference for scale is the watermelon. The expected correct result should show both fruits appearing visually similar in height, width, and overall volume. 3. Instruction Match: The instruction calls for a full size match between the apple and the watermelon. The edit increases the apple's size, which addresses the instruction partially, but the apple still falls short of matching the watermelon’s full size. The core concept is attempted, but not fully realized. The operations in the instruction has been all carried out. 4. Decision: Because the size change was made but not to the full extent required, this counts as 6 partial compliance.",
  "score": 6,
}

## Input
1. **Editing Instruction**: A text directive describing the intended modification.
2. **Image A** — The original image.  
3. **Image B** — The edited image generated according to the instruction.  
4. *(Optional)* **Hint** — Additional textual information that clarifies the **expected correct answer** or the intended editing outcome. Use it as **a reference for scoring accuracy**.  
5. *(Optional)* **Reference Images** — external image that represent the **expected correct result** or serve as additional visual guidance for evaluation.  

## Output Format
Look at the input again, provide the evaluation score and the explanation in the following JSON format:
{
"reason": 1. Detect Difference 2. Expected Visual Caption 3. Instruction Match 4. Decision,
"score": X
}
"""

prompt_visual_quality = """
You are a professional digital artist and image evaluation specialist.

You will be given:
- **Image A**: a single AI-generated image.

## Objective:
Your task is to **evaluate the perceptual quality** of the image, focusing on:
- **Structural and semantic coherence**
- **Natural appearance**
- **Absence of generation artifacts**

You must **not penalize low resolution by image itself**, but you **must** penalize:
- Blur / softness that reduces recognizability or detail,
- Heavy noise or compression artifacts,

## Evaluation Scale (1 to 10):
You will assign a **quality_score** with the following rule:

- **9-10 — Excellent Quality**: All aspects are visually coherent, natural, and free from noticeable artifacts. Structure, layout, and textures are accurate and consistent; Text (if present) is perfectly legible; global clarity and sharpness are high.
- **7-8 — Good (Minor Flaws)**: One small imperfection (e.g., slight texture blending, minor lighting inconsistency, or minor text blur). No major structural or semantic errors.
- **5-6 — Fair (Visible Issues)**: One or two clear visual flaws or semantic problems (e.g., extra fingers, minor duplication, slight distortion). Some areas may appear blurry or noisy, but main content remains understandable.
- **3-4 — Poor (Multiple Errors)**: Multiple distracting errors (e.g., melted hands, warped shapes, unreadable text). Clarity and structure noticeably degraded.
- **1-2 — Fail (Severe Breakdown)**: Major structural failures or hallucinations (e.g., broken anatomy, garbled symbols, text unreadable or nonsensical; strong artifacts or blur dominate).

## Guidance:
Check the following visual aspects and mark them as ✔ (satisfactory) or ✘ (problematic):
- Structural coherence: correct anatomy, object shapes, layout, and spatial relationships.
- Naturalness: plausible lighting, perspective, shading, and shadow logic.
- Artifact-free: no obvious duplication, ghosting, disjoint limbs, seams, or visible watermarks.
- Texture fidelity: clothing, hair, surfaces, and materials are not melted, smudged, or corrupted.
- Optional: Text quality (if text is present), characters are legible, not garbled, with no random symbols; spelling is correct unless obviously stylized or intentionally distorted.
- Optional: Sharpness (only penalize if blur causes semantic loss or very blur!)
✔ The more checks, the higher the score.

Example
{
  "reason": "Structural coherence: ✔, Natural appearance: ✔, Artifacts: ✔, Texture fidelity: ✘ (fabric partially deformed). Text quality: ✘ (random symbols appear, spell wrong). The overall image quality is poor",
  "score": 4
}

## Output Format:
After evaluation, provide your score and reasoning using the following JSON format:
{
"reason": XXX,
"score": X,
}
"""

prompt_knowledge_fidelity_single = """
You are a professional scientific illustrator and image evaluation specialist.

You will be given:
1. **Image A**: the original image.
2. **Image B**: an edited version of Image A.
3. **Editing Instruction**: a directive describing the intended modification to Image A to produce Image B.
4. *(Optional)* **Hint / Reference** — Textual hints and/or reference images that depict or describe what the **knowledge-consistent result** should look like, or what real-world behavior or cultural context is expected.

## Knowledge Plausibility (Guided by Hint / Reference)
Your objective:
Evaluate whether the edited image **accurately reflects real-world logic and knowledge**, including **physical, biological, chemical, cultural, and commonsense correctness**, using the provided hint/reference whenever available.

You must:
- Treat the hint and reference image as **strong guidance** for what the correct, knowledge-aligned result should look like.  
- Judge whether the edited image respects **physical and logical consistency** (e.g., gravity, cause-effect, anatomy, reflection).  
- Assess **conceptual, cultural, and commonsense understanding**, Can the edited images accurately reflect the knowledge in Editing Instruction? — including correct representation of 
  (a) **symbolic or traditional knowledge** (e.g., “Mid-Autumn Festival → mooncake”),  
  (b) **natural and scientific rules** (e.g., “a bamboo shoot grows into bamboo”, “water freezes into ice”), and  
  (c) **cause-effect or behavioral logic** (e.g., “adding weight makes a scale tilt”, “a lit candle produces light”).
  (d) **other type commonsense knowledge**
- Focus strictly on **knowledge plausibility**, not on style or artistry.  
- If no hint/reference is given, infer realism using general world knowledge to judge the edited image.


## Evaluation Scale (1–10):
- **9–10 — Fully Plausible:** Entirely follows real-world or cultural logic, consistent with hint/reference (e.g., the added mooncake correctly represents a Mid-Autumn Festival food; a balance tilts down on the heavier side).  
- **7–8 — Mostly Plausible:** Minor inaccuracies but still realistic and knowledge-aligned.  
- **5–6 — Partially Plausible:** Some correct reasoning but with clear factual or conceptual mistakes.  
- **3–4 — Implausible:** Noticeable conflict with physical or cultural logic.  
- **1–2 — Impossible:** Violates fundamental scientific or commonsense principles (e.g., defies gravity, misrepresents key cultural symbol).

## Example 1 — Physical Logic:
**Instruction:** Add a metal block to the left pan of the scale.  
**Hint:** A correct result should show the left pan tilting downward when weight is added.  
**Observation:** The block is added, but the scale remains level — defying gravity.  
**Output:**
{
  "reason": "After adding a heavy metal block to the left pan, the scale should tilt downward on that side according to gravity and basic physics. However, the edited image keeps the scale balanced, contradicting cause-effect logic.",
  "score": 2
}

## Example 2 — Cultural Knowledge:
**Instruction:** Add a traditional food commonly eaten during the Mid-Autumn Festival on the table.  
**Hint:** The correct result should depict a mooncake on the table.  
**Observation:** The edited image clearly shows a mooncake — round shape, patterned surface, golden-brown color — on the table, matching cultural expectation.  
**Output:**
{
  "reason": "The image correctly adds a mooncake on the table, which aligns with cultural knowledge of the Mid-Autumn Festival. The depiction is conceptually accurate and fully plausible.",
  "score": 9
}

## Output Format
Provide a concise reasoning paragraph describing whether the edited image is consistent with physical, logical, and cultural knowledge.  
Then output your evaluation strictly in the following JSON format:

{
  "reason": "concise reasoning of plausibility judgment",
  "score": X
}
"""

prompt_creative_fusion_single = """
You are a professional art critic and creative image evaluation specialist.

You will be given:
1. **Image A**: the original image.
2. **Image B**: an edited version of Image A.
3. **Editing Instruction**: a directive describing the intended modification to Image A to produce Image B.

## Creativity Evaluation
Your Objective:
Evaluate the **creativity, originality, and imaginative transformation** of the edited image (Image B) relative to the original (Image A) and the given instruction.

You must:
- Focus on **conceptual novelty**, **expressive transformation**, and **visual imagination**.  
- Judge how inventively the edit interprets or reimagines the instruction, especially in **abstract, humorous, or stylistically reinterpreted tasks** such as meme creation, person-to-figurine transformation, or surreal conceptual blending.  
- Consider whether the edit displays **human-like creative thinking**, such as metaphorical reasoning, humor, symbolic representation, or cross-domain imagination. 
- Judge whether the elements on the screen blend harmoniously and be creative or not. 
- Ignore technical imperfections — creativity is about **idea originality**, not visual realism.

## Reasoning Guidance:
When evaluating, consider:
- **Idea Novelty** — Does the result demonstrate an unexpected or unique conceptual leap beyond a literal edit?  
- **Artistic Interpretation** — Does it express imagination through composition, color, or exaggerated style?  
- **Conceptual Depth** — Does it convey humor, emotion, or cultural meaning (e.g., in memes or stylized art)?  
- **Cross-Domain Reimagining** — Does it creatively translate between styles or ontologies (e.g., human → toy, object → living form)?  
- **Coherence** — Even imaginative results should remain visually and semantically coherent.

## Evaluation Scale (1–10):
- **9–10 — Highly Creative:** Exceptional imagination and conceptual depth. The edit shows strong originality, clever reinterpretation, or symbolic transformation far beyond literal instruction-following.  
- **7–8 — Moderately Creative:** Noticeable creativity with some novel elements, but partially conventional or limited in conceptual scope.  
- **5–6 — Mildly Creative:** Some imagination visible, but mostly a direct or expected transformation.  
- **3–4 — Low Creativity:** Simple, mechanical execution of the instruction with minimal conceptual novelty.  
- **1–2 — Non-Creative:** Direct, literal edit; no evidence of imaginative thought.

## Example:
**Editing Instruction:** “Design a creative logo for the word ‘RUN’.”

- ✔ A highly creative version might integrate the letter shapes into a dynamic running figure — for example, transforming the “R” into a bent leg and the “U” into a track curve, conveying motion and energy through typography and composition.  
- ✘ In contrast, the edited image simply adds a small stick figure running beside the unchanged word “RUN.” While this technically follows the instruction, it lacks conceptual imagination or design innovation.

Example output:
{
  "reason": "The image fulfills the literal instruction by adding a small running figure next to the word ‘RUN,’ but the design shows limited creative thinking. The typography remains untouched, and the added figure does not meaningfully integrate with the letters or express motion conceptually. The result feels straightforward and lacks artistic reinterpretation, demonstrating only minimal creativity.",
  "score": 4
}

## Input
1. **Editing Instruction**  
2. **Image A** — The original image.  
3. **Image B** — The edited image generated according to the instruction.

## Output Format
Provide a natural reasoning paragraph analyzing the creativity and imaginative transformation of the edited image — the explanation may be brief or detailed as appropriate.  
Then, output your evaluation in the following JSON format:

{
  "reason": "reasoning about creativity and imagination",
  "score": X
}
"""



