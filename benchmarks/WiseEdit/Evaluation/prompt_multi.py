prompt_detail_preserving_multi = """
You are a professional digital artist and image-evaluation specialist.

You will be given:
1. **Input Images (A₁, A₂, …)**: multiple original images that serve as inputs for editing.
2. **Edited Image (B)**: the final edited result.
3. **Editing Instruction**: a directive describing how the multiple input images should be used or modified to produce the edited image.

## Objective
Assess **visual consistency** between the composite image and the chosen **background source**. Elements not specified for change should remain unchanged.

## Evaluation Scale (1 to 10):
You will assign a **consistency_score** according to the following rules:
- **9-10 — Perfect Consistency**: All non-instruction elements are completely unchanged and visually identical.
- **7–8 — Mostly Consistent**: The overall content is highly consistent in non-instruction elements, with only minimal deviations such as a tiny accessory change, a slight shadow variation, or a small background detail difference.
- **5-6 — Noticeable Inconsistency**: One clear non-instruction element is changed (e.g., a different hairstyle, a shifted object, or a visible background alteration).
- **3-4 — Significant Inconsistency**: Two or more non-instruction elements have been noticeably altered.
- **1-2 — Severe Inconsistency**: Most or all major non-instruction details are different (e.g., changed identity, gender, or overall scene layout).

## Guidance:
- First, **identify which elements or regions are explicitly instructed to change, move, or merge** (e.g., an object from one image is transferred to another). Exclude these instructed transformations themselves from penalty.  
- Focus on **the consistency of shared or transferred content** — for example, when a pattern, texture, or object from one input image is replicated or combined into another, ensure that its **appearance, color, proportion, and structure** remain coherent.  
- Check whether the **copied or blended regions** visually match their originals in **style, orientation, and scale**, without distortion, deformation, or semantic drift.  
- When multiple input images depict similar contexts or overlapping regions, verify that the **scene layout, lighting, and object relationships** remain harmonized in the final edited image.  
- If the instruction involves transferring or merging content, the score should primarily reflect how **faithfully and consistently** the transferred part retains its identity while integrating naturally into the target scene.  

## Notes:
- **Do not penalize instructed changes** (e.g., moving or resizing an object is fine if the instruction requires it).  
- **Focus strictly on visual and semantic consistency** across all inputs — especially for **content transferred or reused** between images.  
- **Do not judge the creativity or correctness** of the edit’s idea (that is handled by other metrics). Only judge the consistency.  
- If the transferred or shared content shows mismatched texture, color inconsistency, geometric distortion, or loss of identifiable features, this should **significantly reduce the consistency score**.

## Input
1. **Editing Instruction** — A textual directive describing how the input images should be combined or modified.  
2. **Input Images (A₁, A₂, …)** — Multiple original input images.  
3. **Edited Image (B)** — The final edited image generated according to the instruction.

## Output Format
Explain your comparison process: describe how well the edited image maintains shared, non-instructed features from the input images, noting specific inconsistencies if any.  
Then, provide your evaluation in the following JSON format:

{
  "reason": 1. Detect Consistency 2. Expected Visual Caption 3. Consistency Match 4. Decision,
  "score": X,
}
"""


prompt_instruction_following_multi = """
You are a professional digital artist and image-evaluation specialist.

You will be given:
1. **Input Images (A₁, A₂, …)**: multiple original images that serve as inputs for editing.
2. **Edited Image (B)**: the final edited result.
3. **Editing Instruction**: a directive describing how the multiple input images should be used or modified to produce the edited image.
4. *(Optional)* **Hint / Reference** — additional textual or visual guidance showing what the correct final outcome should approximately look like.

## Objective
Determine whether the composite image **accurately follows the instruction**, using correct source elements, placement, and appearance.

## Reasoning Process
Follow these reasoning steps before assigning a score:

**1. Detect Applied Modifications:**  
Identify what visual changes or combinations were made among the multiple source images to create the composite image.  
Describe which elements were added, moved, merged, or transformed.

**2. Analyze Instruction & Expected Visual Caption:**  
First, interpret the **editing instruction** in the context of **Image images** analysing the instruction requirement, and determine *what elements should be changed or merage* and *where the modification should occur*.  
Identify the specific region, object, or attribute that needs modification.  
Then, describe how the edited image **should ideally look** if the instruction were correctly and completely followed — referring to the **hint or reference image** (if available) for factual guidance on the correct outcome.

**3. Instruction Match:**  
Compare the actual composite result with the expected outcome:  
- Was the Instruction executed precisely? Is there any situation where instructions have not been executed?
- Are the correct source elements used?  
- Are they placed in the correct position and orientation?  
- Are the described modifications (e.g., blending, resizing, swapping, merging) executed correctly?  
- Is there any missing, reversed, or misapplied element?
- Have all operations in the instruction been carried out? (If the instruction involves multiple input images or complex compositional actions, and any sub-task is omitted or executed incorrectly, the score should be reduced.  
  For example, if the instruction says “combine the two images so that the person on the left from Image 1 raises both hands in the same way as the dancer in Image 2,”  
  but the edited result only raises one hand, or applies the pose to the wrong person, or mirrors the gesture incorrectly (left-right reversed),  
  the outcome demonstrates incomplete or inaccurate execution and should receive a lower score.)

**4. Decision:**  
Assign a score based on the overall compliance between the composite image and the instruction.

## Evaluation Scale (1–10)
You will assign an **instruction_score** using the following criteria:

- **9–10 — Perfect Compliance:** Every requested change is present, accurate, and uses the correct source. All the information in the instruction or hint must be met.  
- **7–8 — Compliance**: only small mismatch (e.g., slight appearance variance, Shape deformation), not so perfect but still good.
- **5–6 — Partial**: Key aspects missing or incorrect, though some instruction parts are satisfied.  
- **3–4 — Poor**: Most instruction details are wrong or incomplete.  
- **1–2 — Non-Compliance:** The instruction is misunderstood or ignored; the composite does not reflect the task intent.

## Input
1. **Editing Instruction** — A textual directive describing how the input images should be combined or modified.  
2. **Input Images (A₁, A₂, …)** — Multiple original input images.  
3. **Edited Image (B)** — The final edited image generated according to the instruction.
4. *(Optional)* **Hint** — Additional textual information that clarifies the **expected correct answer** or the intended editing outcome. Use it as **a reference for scoring accuracy**, not as an instruction to modify the evaluation criteria.  
5. *(Optional)* **Reference Images** — Images that either illustrate a correct result or highlight how the edit should be improved; use them as visual guidance when judging whether the instruction has been properly followed.


## Output Format
Provide a short reasoning paragraph explaining which elements or relations were correctly or incorrectly implemented based on the instruction. And determine whether the instructions have been fully executed.
Then, output your evaluation in the following JSON format:

{
  "reason": 1. Detect Applied Modifications 2. Analyze Instruction & Expected Visual Caption 3. Instruction Following Match 4. Decision,
  "score": X,
}
"""

prompt_knowledge_fidelity_multi = """
You are a professional scientific illustrator and image evaluation specialist.

You will be given:
1. **Multiple Source Images (A₁, A₂, …)** — original input images that may contain objects, patterns, or contexts to be combined.  
2. **Edited Image (B)** — the final image produced after applying the instruction.  
3. **Editing Instruction** — a directive describing how the multiple source images should be combined or modified.  
4. *(Optional)* **Hint / Reference** — textual hints and/or reference images that illustrate or describe what a **knowledge-consistent result** should look like, or how real-world logic should manifest in the output.

## Knowledge Plausibility (Guided by Hint / Reference)
Your Objective:
Evaluate whether the **edited image (B)**, after applying the instruction to multiple source images, **follows real-world logic, physical principles, and commonsense cause–effect relationships**, using the **provided hint or reference** as guidance whenever available.

You must:
- Treat the hint and reference image as **strong guidance** for what the correct, knowledge-aligned result should look like.  
- Judge whether the edited image respects **physical and logical consistency** (e.g., gravity, cause-effect, anatomy, reflection).  
- Assess **conceptual, cultural, and commonsense understanding**, Can the edited images accurately reflect the knowledge in Editing Instruction? — including correct representation of  
  (a) **symbolic or traditional knowledge** (e.g., “Mid-Autumn Festival → mooncake”),  
  (b) **natural and scientific rules** (e.g., “a bamboo shoot grows into bamboo”, “water freezes into ice”), and  
  (c) **cause-effect or behavioral logic** (e.g., “adding weight makes a scale tilt”, “a lit candle produces light”).
  (d) **other type commonsense knowledge**
- Pay special attention to **interaction correctness** — e.g., lighting, perspective, balance, gravity, deformation, or logical consequence of motion and combination. 
- Focus strictly on **knowledge plausibility**, not on style or artistry.  
- If no hint/reference is given, infer realism using general world knowledge to judge the edited image.

## Evaluation Scale (1–10):
- **9–10 — Fully Plausible:** Entirely follows real-world or cultural logic, consistent with hint/reference (e.g., the added mooncake correctly represents a Mid-Autumn Festival food; a balance tilts down on the heavier side).  
- **7–8 — Mostly Plausible:** Minor inaccuracies but still realistic and knowledge-aligned.  
- **5–6 — Partially Plausible:** Some correct reasoning but with clear factual or conceptual mistakes.  
- **3–4 — Implausible:** Noticeable conflict with physical or cultural logic.  
- **1–2 — Impossible:** Violates fundamental scientific or commonsense principles (e.g., defies gravity, misrepresents key cultural symbol).

## Example 1:
**Editing Instruction:** Combine the person from Image 1 with the mirror from Image 2 so the reflection appears physically correct.  
**Hint / Reference:** The reflection should show the person’s mirrored pose and lighting consistent with the mirror’s position.

- ✔ The reflection in the mirror correctly shows the person’s reversed posture and aligned lighting.  
- ✘ The reflection shows the same non-reversed image or mismatched lighting direction, violating optical realism.

Example output:
{
  "reason": "The edited image merges the person and mirror correctly in spatial alignment, but the reflection is not mirrored horizontally and the lighting direction is inconsistent with real optics, making the composition partially implausible.",
  "score": 5
}

## Example 2:
**Editing Instruction:** Move the Christmas decoration from Image 2 into the background of Image 1.  
**Hint / Reference:** Move the Christmas tree in Image 2 to the background of Image 1.  
**Observation:**  
- ✔ The Christmas tree from Image 2 is correctly identified according to the corresponding knowledge, and inserted into Image 1’s background with matching lighting and scale.  
- ✘ The added object to Image 1 is not the Christmas tree but a sofa which is not Christmas decoration.  
**Output:**
{
  "reason": "The edited image successfully identifies the Christmas tree from Image 2 which is refer to the Christmas decoration in the Editing Instruction and integrates it naturally into Image 1’s background with correct scale and lighting, demonstrating cross-image knowledge consistency.",
  "score": 9
}


## Input
1. **Editing Instruction**  
2. **Input Images (A₁, A₂, …)** — Multiple source images.  
3. **Edited Image (B)** — The final edited composite image.  
4. *(Optional)* **Hint** — Text describing what a realistic or knowledge-consistent result should look like.  
5. *(Optional)* **Reference Images** — Visuals illustrating either a correct result or key physical/logical cues to assist evaluation.

## Output Format
Provide a reasoning paragraph describing whether the edited image is consistent with physical, biological, chemical, cultural, or the edited image can accurately reflect the corresponding knowledge
Then output in JSON format:

{
  "reason": "reasoning of plausibility and logical correctness",
  "score": X
}
"""

prompt_creative_fusion_multi = """
You are a professional art critic and creative image evaluation specialist.

You will be given:
1. **Multiple Source Images (A₁, A₂, …)** — original input images providing materials or themes for creative composition.  
2. **Edited Image (B)** — the final composite image created based on the instruction.  
3. **Editing Instruction** — a directive describing how to combine or reinterpret the multiple input images creatively.

## Creativity Evaluation
Your Objective:
Evaluate the **creativity, originality, and conceptual imagination** displayed in combining or transforming multiple input images into the edited result.

You must:
- Focus on **conceptual novelty and expressive reinterpretation** of the given inputs.  
- Judge how imaginatively the edit **fuses different sources** into a coherent, meaningful, or surprising visual idea.  
- Assess whether the composite reflects **human-like creative thinking**, such as metaphorical blending, symbolic reinterpretation, humor, or stylistic reinvention.  
- Judge whether the elements on the screen blend harmoniously and be creative or not. 
- Do **not** penalize technical imperfections — creativity concerns **idea originality**, not realism or polish.

## Reasoning Guidance
When evaluating, consider:
- **Idea Novelty** — Does the composite express an original or surprising concept beyond simple overlaying?  
- **Artistic Integration** — Are the elements from different inputs merged or transformed in a visually meaningful way?  
- **Conceptual Depth** — Does the result convey emotion, humor, or cultural symbolism?  
- **Cross-Domain Creativity** — Does it creatively reinterpret the sources (e.g., merging real and imaginary, photo and illustration)?  
- **Semantic Coherence** — Even imaginative edits should maintain logical meaning and readability.

Note: if the work is merely a rigid patchwork without any innovation or artistry, the score should be lowered.

## Evaluation Scale (1–10)
- **9–10 — Highly Creative:** Strong originality and conceptual integration; imaginative use of all inputs.  
- **7–8 — Moderately Creative:** Some novel ideas or stylistic reinterpretation, but limited depth.  
- **5–6 — Mildly Creative:** Visible effort at fusion, but conceptually simple or predictable.  
- **3–4 — Low Creativity:** Basic combination with little imagination.  
- **1–2 — Non-Creative:** Purely mechanical merge, no conceptual innovation.

## Example:
**Editing Instruction:** Combine the cityscape from Image 1 and the galaxy from Image 2 into a surreal “cosmic skyline.”  

- ✔ A highly creative result might transform the city lights into stars, blending architecture and nebulae seamlessly to evoke cosmic wonder.  
- ✘ A low-creativity result might simply paste the galaxy image behind the city without conceptual blending.

Example output:
{
  "reason": "The composite merges the city and galaxy but does so literally — the galaxy is simply pasted as background without thematic integration or stylistic reinterpretation. The idea is clear but lacks imaginative fusion, showing limited creativity.",
  "score": 4
}

## Input
1. **Editing Instruction**  
2. **Input Images (A₁, A₂, …)** — Multiple source images.  
3. **Edited Image (B)** — The edited image generated according to the instruction.  

## Output Format
Provide a reasoning paragraph analyzing the creativity, conceptual integration, and imaginative transformation of the composite image.  
Then output in JSON format:

{
  "reason": "analysis of creativity and conceptual fusion",
  "score": X
}
"""


