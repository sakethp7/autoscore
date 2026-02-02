from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
from typing import List, Optional
import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field
from PIL import Image
import base64
from io import BytesIO
import asyncio
from contextlib import asynccontextmanager
import logging
import uuid
from threading import Lock
import copy
# Load environment variables
load_dotenv()
# tested
# --- Configuration & Initialization ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Environment variables for configuration
MAX_LLM_CONNECTIONS = int(os.getenv('MAX_LLM_CONNECTIONS', '10'))
API_KEY = os.getenv('GEMINI_API_KEY')

if not API_KEY:
    logger.error("GOOGLE_API_KEY environment variable not set.")
    raise ValueError("GOOGLE_API_KEY environment variable not set")

# --- System Prompts with Rubrics ---
EVALUATION_SYSTEM_PROMPT = """You are an expert teacher and evaluator for student homework and exam answers. You must evaluate answers with strict consistency and fairness.

## SCORING RUBRIC (MANDATORY - APPLY EXACTLY)

### For 2-mark questions:
- 1 mark: Correct concept/formula used
- 1 mark: Correct final answer

### For 5-mark questions:
- 2 marks: Correct concept/formula used
- 2 marks: Correct approach/methodology/steps
- 1 mark: Correct final answer

### For 10-mark questions:
- 4 marks: Correct concept/formula used
- 4 marks: Correct approach/methodology/steps
- 2 marks: Correct final answer

## DEDUCTION RULES (STRICT)
- Wrong concept → 0 marks for concept section, method and answer marks also become 0
- Correct concept, wrong method → Award concept marks only, 0 for method and answer
- Correct concept and method, calculation error → Award concept + method marks, deduct from answer marks only
- Missing units (when required) → Deduct max 0.5 mark from final answer section only
- Irrelevant answer → 0 marks total
- NEVER deduct twice for the same mistake
- Partial credit: If concept is correct but method and answer are wrong, award ONLY concept marks

## Please understand formulaes and concepts deeply to evaluate properly.
## ERROR TYPE CLASSIFICATION (Choose exactly ONE)
- Conceptual Error: Wrong concept, formula, or theory applied
- Calculation Error: Arithmetic or computational mistakes with correct concept/method
- Logical Error: Mistake in reading values from question or flawed reasoning sequence
- Irrelevant: Answer completely unrelated to the question
- None: No errors, fully correct answer

## TIME MANAGEMENT RUBRIC
Based on expected time vs actual time taken:
- If max_time = T minutes:
  - Student completes in < (T - 2) minutes → "great" (excellent time management)
  - Student completes in (T - 2) to T minutes → "good" (within expected time)
  - Student completes in T to (T + 2) minutes → "should_improve" (slightly over time)
  - Student completes in > (T + 2) minutes → "critical" (significant time issues)

Example for 5-minute question:
- < 3 minutes → great
- 3-5 minutes → good
- 5-7 minutes → should_improve
- > 7 minutes → critical

## EVALUATION METHODOLOGY
You must follow Chain-of-Thought reasoning for every evaluation:
1. First, identify the question requirements and expected solution approach
2. Then, analyze what concept/formula the student used
3. Next, trace through the student's methodology step-by-step
4. Check the final answer against the expected result
5. Classify any errors found
6. Apply the scoring rubric systematically
7. Provide constructive feedback
8. Provide detaied score breakdwn whya you have cut the marks and reasons

## Gap Analysis:
- Generate a single-line gap analysis that clearly states the exact missing or weak concept(s) of the student.
- The output must be concise, specific, and actionable, avoiding explanations, examples, or extra details.
- Focus only on what the student does not understand, not what they did correctly.

## Latex Rules
Convert ALL mathematical expressions using these rules:

            **Inline Math:** Convert `$...$` or `\\(...\\)` → `$...$`
            **Display Math:** Convert `$$...$$` or `\\[...\\]` → `$$...$$`

            **Common Conversions:**
            - `\\frac(a)(b)` → `\\frac(a)(b)` (keep same)
            - `\\int` → `\\int` (keep same)
            - `\\sum` → `\\sum` (keep same)
            - `\\sqrt(x)` → `\\sqrt(x)` (keep same)
            - `x^(2)` → `x^2`
            - `x_(1)` → `x_1`
            - `\\sin(x)`, `\\cos(x)`, `\\log(x)` → keep same
            - `\\theta`, `\\alpha`, `\\beta` → keep same
            - `\\cdot` → `\\cdot` or `·` (keep multiplication symbols)
            - `\\times` → `\\times` (keep same)
            - `\\pm` → `\\pm` (keep same)
            - `\\infty` → `\\infty` (keep same)

            **Matrix/Array Conversions:**
            - Convert `\\begin(array)` → `\\begin(array)` (KaTeX supports this)
            - Convert `\\begin(matrix)` → `\\begin(matrix)` (KaTeX supports this)


Example:
$ \frac{\sqrt{3}}{4}a^2$

## OUTPUT REQUIREMENTS
- Be consistent: Same type of answer should always get same score
- Be fair: Award partial credit where deserved
- Be specific: Clearly identify what was right and wrong
- Be constructive: Provide actionable feedback for improvement
"""

HOMEWORK_EVALUATION_SYSTEM_PROMPT = """You are an expert teacher evaluating student homework submissions. You must evaluate ALL questions thoroughly and consistently.

## SCORING RUBRIC (MANDATORY - APPLY EXACTLY)

### Mark Distribution by Question Type:
- Conceptual Understanding: 50% of total marks
- Problem-solving Procedure: 20% of total marks
- Final Answer Accuracy: 10% of total marks
- Mathematical Methods/Formulas: 20% of total marks

### For sub-questions:
- Divide total marks EQUALLY among sub-parts
- Example: Q2 (6 marks) with 3 sub-parts = 2 marks each for (a), (b), (c)

## ERROR TYPE CLASSIFICATION (Choose exactly ONE per question)
- conceptual_error: Wrong concept or theory applied
- calculation_error: Arithmetic or computational mistakes
- logical_error: Flawed reasoning or incorrect sequence
- no_error: Completely correct answer
- unattempted: Question not attempted

## TIME MANAGEMENT RUBRIC
For each question based on expected time vs actual performance:
- great: Completed well under expected time with correct answer
- good: Completed within expected time
- should_improve: Took longer than expected
- critical: Significantly exceeded time or incomplete due to time

## EVALUATION METHODOLOGY (Chain-of-Thought Required)
For EACH question, you must:
1. IDENTIFY: What is the question asking? What concepts are needed?
2. ANALYZE: What did the student write? What approach did they use?
3. COMPARE: How does their solution compare to the expected approach?
4. TRACE: Go through their work step-by-step looking for errors
5. CLASSIFY: What type of error (if any) did they make?
6. SCORE: Apply the rubric systematically
7. FEEDBACK: What specific improvements are needed?

## IMPORTANT RULES
- Only evaluate questions from the original question paper
- Match student answers to correct question numbers
- For unattempted questions: score=0, mistakes_made="Question not attempted"
- Never exceed maximum marks for any section
- Be consistent across all questions
- Provide specific, actionable feedback
"""

# --- Pydantic Models ---

class QuestionEvaluation(BaseModel):
    """Evaluation result for a single question"""
    question_number: str = Field(description="Question number or identifier")
    question_text: str = Field(description="The question that was evaluated")
    score: int = Field(description="Score for the student's answer")
    max_marks: int = Field(description="Maximum marks for the question")
    error_type: str = Field(description="One of error: Conceptual Error,Irrelavant, Calculation Error or None")
    mistakes_made: str = Field(description="Specific mistakes made in the question")
    gap_analysis: str = Field(description="Detailed explanation about student's mistakes and missing concepts")
    additional_comments: str = Field(description="Any additional comments")
    concepts_required: str = Field(description=" 2-3 Main concepts required to solve the question")
    time_analysis: str = Field(description="Time analysis: great/good/should_improve/critical")

class HomeworkEvaluationResult(BaseModel):
    """Complete homework evaluation result containing all questions"""
    total_questions: int = Field(description="Total number of questions evaluated")
    total_score: int = Field(description="Total score obtained across all questions")
    total_max_marks: int = Field(description="Total maximum marks for all questions")
    evaluations: List[QuestionEvaluation] = Field(description="List of individual question evaluations")
    overall_performance: str = Field(description="Overall performance summary")
    extracted_text_summary: str = Field(description="Summary of what was extracted from answer sheets")

class Result(BaseModel):
    """Evaluation Result schema for a single question"""
    score: int = Field(description="Score for the student's answer")
    max_marks: int = Field(description="Maximum marks for the question")
    error_type: str = Field(description="Type of error: conceptual_error, calculation_error, logical_error, no_error or unattempted")
    mistakes_made: str = Field(description="Specific mistakes made in the question")
    gap_analysis: str = Field(description="Explain in detail about student's mistakes what concept he is unable to apply to solve the problem")
    additional_comments: str = Field(description="Any additional comments")
    concepts_required: str = Field(description="Concepts required to solve the question")
    time_analysis: str = Field(description="Time analysis in one word great/good/should improve/critical")
    score_breakdown: str = Field( description="detailed score breakdown")
# --- FIXED LLM Connection Pool ---
class LLMPool:
    """Thread-safe connection pool for LLM instances with proper isolation."""

    def __init__(self, size: int):
        self.size = size
        self.semaphore = asyncio.Semaphore(self.size)
        self._lock = asyncio.Lock()

    def _create_llm_instance(self) -> ChatGoogleGenerativeAI:
        """Creates a fresh LLM instance (gemini-2.5-flash) with optimized parameters."""
        return ChatGoogleGenerativeAI(
            model='gemini-2.5-flash',
            api_key=API_KEY,
            temperature=0,
            top_p=0.1,
            top_k=10,
            max_retries=2,
            timeout=60
        )


    def _create_llm_instance_lite(self) -> ChatGoogleGenerativeAI:
        """Creates a fresh LLM instance (gemini-2.5-flash-lite) with optimized parameters."""
        return ChatGoogleGenerativeAI(
            model='gemini-2.5-flash',
            api_key=API_KEY,
            temperature=0,
            top_p=0.1,
            top_k=10,
            max_retries=2,
            timeout=60
        )


    @asynccontextmanager
    async def get_llm(self):
        """
        Provides a fresh 'gemini-2.5-flash' instance for each request.
        """
        await self.semaphore.acquire()
        try:
            llm = self._create_llm_instance()
            yield llm
        except Exception as e:
            logger.error(f"Error creating LLM instance: {e}")
            raise
        finally:
            self.semaphore.release()

    @asynccontextmanager
    async def get_llm_lite(self):
        """
        Provides a fresh 'gemini-2.5-flash-lite' instance for each request.
        """
        await self.semaphore.acquire()
        try:
            llm = self._create_llm_instance_lite()
            yield llm
        except Exception as e:
            logger.error(f"Error creating LLM-Lite instance: {e}")
            raise
        finally:
            self.semaphore.release()

# Global LLM pool instance
llm_pool = LLMPool(size=MAX_LLM_CONNECTIONS)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manages application startup and shutdown lifecycle."""
    logger.info(f"FastAPI app starting with {MAX_LLM_CONNECTIONS} LLM connections.")
    yield
    logger.info("FastAPI app shutting down.")

app = FastAPI(
    title="AutoScore API",
    description="Student Answer Evaluation System",
    version="3.1.0",
    lifespan=lifespan
)

# --- FIXED Utility Functions ---
async def image_to_base64_async(image: Image.Image, request_id: str) -> str:
    """Converts a PIL image to a base64 string asynchronously with proper isolation."""
    loop = asyncio.get_event_loop()
    # Create a copy of the image to prevent concurrent modification
    image_copy = image.copy()
    return await loop.run_in_executor(None, image_to_base64_sync, image_copy, request_id)

def image_to_base64_sync(image: Image.Image, request_id: str) -> str:
    """Blocking function to convert and optimize a PIL image copy."""
    try:
        # Work on the copied image to prevent concurrent modification issues
        buffered = BytesIO()
        max_size = (1920, 1920)

        if image.size[0] > max_size[0] or image.size[1] > max_size[1]:
            image.thumbnail(max_size, Image.Resampling.LANCZOS)

        if image.mode == 'RGBA':
            background = Image.new('RGB', image.size, (255, 255, 255))
            background.paste(image, mask=image.split()[3])
            image = background

        image.save(buffered, format="JPEG", quality=85, optimize=True)
        encoded = base64.b64encode(buffered.getvalue()).decode()

        logger.debug(f"[{request_id}] Image converted to base64, size: {len(encoded)}")
        return encoded

    except Exception as e:
        logger.error(f"[{request_id}] Error converting image to base64: {e}")
        raise

async def extract_text_from_image(llm: ChatGoogleGenerativeAI, image: Image.Image, page_num: int, request_id: str) -> str:
    """Phase 1: Extracts text from image using OCR via LLM (async)."""
    ocr_prompt = """Extract ALL text from this image with extreme precision for student answer evaluation.

## STEP-BY-STEP EXTRACTION PROCESS
Follow these steps in order:

### Step 1: Scan the entire image
- Identify all question numbers and their locations
- Note any crossed-out sections to skip

### Step 2: For each question, extract in order:
- Question number/identifier
- All written content
- Mathematical expressions
- Diagrams/figures descriptions

### Step 3: Format the extraction

**EXTRACTION RULES:**

1. **QUESTION STRUCTURE:**
   - Main questions: "1)", "2)", "Q1", "Question 1", etc.
   - Sub-questions: "(i)", "(ii)", "(iii)", "(a)", "(b)", "(c)", "Part A", "Part B", etc.

2. **MATHEMATICAL CONTENT:**
   - Use LaTeX notation for ALL mathematical expressions
   - All LaTeX must be **KaTeX-compatible** and properly escaped for use in `react-katex`
   - Inline math: $expression$
   - Display math: $$expression$$
   - Preserve ALL steps, calculations, and working

3. **TEXT CONTENT:**
   - Preserve ALL written explanations
   - Include margin notes, corrections, and annotations

4. **VISUAL ELEMENTS:**
   - Describe diagrams: [DIAGRAM: description]
   - Describe graphs: [GRAPH: axes labels, curves, points]
   - Describe tables: [TABLE: structure and content]
   - Describe geometric figures: [FIGURE: shape, labels, measurements]

5. **CROSSING OUT RULES**
   - If student crosses out something don't retrieve that. This includes for all figures, tables, diagrams etc.
   - If student crossed out using big X for text or diagram entirely don't mention that in extracted content
   - If student simplyfied  any equation for simplifying, dont consider that as crossed out that is simplyfication
   - You can leave the text, figure or diagram if it is crossed out using X.

6. **FORMATTING:**
   - Maintain original structure and indentation
   - Preserve bullet points, numbering, and lists
   - Keep line breaks where significant

**OUTPUT FORMAT:**
[Question Number])
[Student's Solution:]
[All work, steps, calculations in exact order]
[Final answer if marked]
[Continue for all questions on page...]

**START EXTRACTION NOW:**"""

    try:
        img_base64 = await image_to_base64_async(image, request_id)
        message = [{
            "role": "user",
            "content": [
                {"type": "text", "text": ocr_prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"}}
            ]
        }]

        logger.info(f"[{request_id}] Extracting text from page {page_num + 1}")
        response = await llm.ainvoke(message)
        extracted_text = response.content if hasattr(response, 'content') else str(response)

        # Simple text cleaning
        cleaned_text = '\n'.join([
            line for line in extracted_text.split('\n')
            if not line.strip().startswith(('Roll:', 'Page:'))
        ])

        result = f"<page{page_num + 1}>\n{cleaned_text}\n</page{page_num + 1}>"
        logger.info(f"[{request_id}] Successfully extracted text from page {page_num + 1}")
        return result

    except Exception as e:
        logger.error(f"[{request_id}] Error extracting text from page {page_num + 1}: {e}")
        return f"<page{page_num + 1}>\nError extracting text: {str(e)}\n</page{page_num + 1}>"

async def evaluate_answer(llm: ChatGoogleGenerativeAI,question: str, question_images: list, extracted_answer: str, request_id: str) -> Result:
    """Phase 2: Evaluates the extracted answer (async) with Chain-of-Thought reasoning."""

    evaluation_prompt = f"""## EVALUATION TASK
Evaluate the following student answer using Chain-of-Thought reasoning.

## QUESTION DETAILS
- **Question:** {question}
- **Student's Answer:** {extracted_answer}



- Max marks:5

## CHAIN-OF-THOUGHT EVALUATION (Follow these steps exactly)

### Step 1: Question Analysis
Think: What is this question asking? What concept/formula is expected?

### Step 2: Student Response Analysis
Think: What concept did the student use? Is it the correct concept?

### Step 3: Methodology Check
Think: Trace through the student's steps. Are the steps logical and correct?

### Step 4: Final Answer Verification
Think: What is the student's final answer? Is it correct?

### Step 5: Error Classification
Think: Based on my analysis, what type of error (if any) did the student make?
- If wrong concept → Conceptual Error
- If right concept but calculation mistake → Calculation Error
- If misread question values → Logical Error
- If answer unrelated to question → Irrelevant
- If no errors → None

### Step 6: Score Calculation
Think: Apply the scoring rubric from system instructions:
- Award marks for correct concept
- Award marks for correct method
- Award marks for correct answer
- Apply deductions appropriately

### Step 7: Time Assessment
Think: Based on the complexity and student's work shown:
- Simple and correct → great
- Appropriate effort → good
- Shows struggle/excessive work → should_improve
- Incomplete or very messy → critical

## OUTPUT REQUIREMENTS
Based on your chain-of-thought analysis above, provide:
- score: Calculated score (0 to max_marks)
- max_marks: Total marks for this question
- error_type: One of - Conceptual Error | Irrelevant | Calculation Error | Logical Error | None
- mistakes_made: Specific mistakes identified in Step 3-4
- gap_analysis: Detailed explanation of what concepts the student doesn't understand
- additional_comments: Constructive feedback for improvement
- concepts_required: 2-3 key topic names needed to solve this question
- time_analysis: One of - great | good | should_improve | critical

Remember: Do not use underscores in error_type (use "Conceptual Error" not "conceptual_error")
"""

    try:
        # Build message with system prompt
        system_message = {"role": "system", "content": EVALUATION_SYSTEM_PROMPT}

        if question_images:
            message_content = [{"type": "text", "text": evaluation_prompt}]
            for i, img in enumerate(question_images):
                img_base64 = await image_to_base64_async(img, f"{request_id}_q{i}")
                message_content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"}
                })
            messages = [system_message, {"role": "user", "content": message_content}]
        else:
            messages = [system_message, {"role": "user", "content": evaluation_prompt}]

        logger.info(f"[{request_id}] Starting evaluation")
        structured_llm = llm.with_structured_output(Result)
        result = await structured_llm.ainvoke(messages)
        logger.info(result)
        logger.info(f"[{request_id}] Evaluation completed successfully")
        return result

    except Exception as e:
        logger.error(f"[{request_id}] Error in evaluation: {e}")
        raise HTTPException(status_code=500, detail=f"Error in evaluation: {str(e)}")

# --- API Endpoints ---
@app.get("/")
async def root():
    """Root endpoint to check API status."""
    return {
        "message": "AutoScore API is running",
        "version": "2.0.0",
        "max_connections": llm_pool.size,
        "available_connections": llm_pool.semaphore._value
    }

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "llm_pool_size": llm_pool.size,
        "available_llm_instances": llm_pool.semaphore._value
    }

async def evaluate_homework_batch(
    llm: ChatGoogleGenerativeAI,
    questions: List[dict],
    question_images: List[Image.Image],
    extracted_answer: str,
    request_id: str
) -> HomeworkEvaluationResult:
    """
    Phase 2: Evaluates all homework questions in a single LLM call using structured output with CoT.
    """

    # Format questions for the prompt
    questions_formatted = "\n\n".join([
        f"Question {q.get('question_number', idx+1)}: {q.get('question_text', q.get('text', ''))}"
        for idx, q in enumerate(questions)
    ])

    evaluation_prompt = f"""## HOMEWORK EVALUATION TASK
Evaluate ALL questions using Chain-of-Thought reasoning for each.

## QUESTIONS TO EVALUATE
{questions_formatted}

**Note:** Each question is for 10 marks. If a question has sub-questions, split marks equally.

## STUDENT'S COMPLETE ANSWER SHEET
{extracted_answer}

## CHAIN-OF-THOUGHT EVALUATION PROCESS

For EACH question, follow these steps:

### Step 1: IDENTIFY
- What is this question asking?
- What concepts/formulas are required?
- What is the expected solution approach?

### Step 2: LOCATE
- Find the student's answer for this question number
- If not found, mark as unattempted

### Step 3: ANALYZE
- What concept did the student apply?
- Is it the correct concept for this question?

### Step 4: TRACE
- Go through each step of the student's work
- Check calculations at each step
- Verify logical flow

### Step 5: COMPARE
- Compare student's final answer to expected answer
- Note any discrepancies

### Step 6: CLASSIFY ERROR
- Based on analysis, determine error type:
  - conceptual_error: Wrong concept/formula
  - calculation_error: Math mistakes with correct concept
  - logical_error: Wrong values or flawed reasoning
  - no_error: Completely correct
  - unattempted: Not answered

### Step 7: SCORE
Apply rubric:
- 50% for conceptual understanding
- 20% for problem-solving procedure
- 10% for final answer accuracy
- 20% for mathematical methods

### Step 8: TIME ASSESSMENT
- great: Clean, efficient solution
- good: Reasonable approach and time
- should_improve: Excessive work or struggling
- critical: Incomplete or very inefficient

## EVALUATION SCOPE RULES
- ONLY evaluate questions from the original question paper
- Match student answers to correct question numbers
- Ignore any extra questions student may have attempted
- For unattempted: score=0, mistakes_made="Question not attempted"

## OUTPUT REQUIREMENTS
Provide complete evaluation with:
- Individual evaluation for each question
- Total score calculation
- Overall performance summary
- Use LaTeX with spaces around symbols (not directly adjacent to periods)
- concepts_required: Be specific, mention only topic names

Think through each question systematically before providing your evaluation.
"""

    try:
        # Build message with system prompt
        system_message = {"role": "system", "content": HOMEWORK_EVALUATION_SYSTEM_PROMPT}

        if question_images:
            message_content = [{"type": "text", "text": evaluation_prompt}]
            for i, img in enumerate(question_images):
                img_base64 = await image_to_base64_async(img, f"{request_id}_qimg{i}")
                message_content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"}
                })
            messages = [system_message, {"role": "user", "content": message_content}]
        else:
            messages = [system_message, {"role": "user", "content": evaluation_prompt}]

        logger.info(f"[{request_id}] Starting batch evaluation for {len(questions)} questions")

        # Use structured output to get all evaluations at once
        structured_llm = llm.with_structured_output(HomeworkEvaluationResult)
        result = await structured_llm.ainvoke(messages)

        logger.info(f"[{request_id}] Batch evaluation completed successfully")
        return result

    except Exception as e:
        logger.error(f"[{request_id}] Error in batch evaluation: {e}")
        raise HTTPException(status_code=500, detail=f"Error in batch evaluation: {str(e)}")


@app.post("/homework-autoscore/")
async def homework_autoscore(
    questions: str = Form(..., description="JSON string containing list of questions"),
    question_images: Optional[List[UploadFile]] = File(None, description="Optional images for questions (map by index)"),
    answer_images: List[UploadFile] = File(..., description="Student answer images containing all homework answers")
):
    """
    Evaluate multiple homework questions from student answer sheets in a single batch.
    """
    # Generate unique request ID
    request_id = str(uuid.uuid4())[:8]
    logger.info(f"[{request_id}] Starting homework autoscore batch request")

    try:
        # Parse and validate questions JSON
        import json
        try:
            questions_data = json.loads(questions)
            if not isinstance(questions_data, list):
                raise ValueError("Questions must be a list")

            # Normalize question format
            normalized_questions = []
            for idx, q in enumerate(questions_data):
                if isinstance(q, str):
                    normalized_questions.append({
                        'question_text': q,
                        'question_number': str(idx + 1)
                    })
                elif isinstance(q, dict):
                    normalized_questions.append({
                        'question_text': q.get('question_text', q.get('text', '')),
                        'question_number': q.get('question_number', str(idx + 1))
                    })
                else:
                    raise ValueError(f"Invalid question format at index {idx}")

        except json.JSONDecodeError as e:
            raise HTTPException(status_code=400, detail=f"Invalid JSON in questions: {str(e)}")
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

        if not normalized_questions:
            raise HTTPException(status_code=400, detail="No questions provided")

        logger.info(f"[{request_id}] Processing batch of {len(normalized_questions)} questions")

        # Process all images concurrently
        question_image_objects = []
        if question_images:
            question_image_tasks = [
                read_and_process_image(img_file, f"{request_id}_qimg{i}")
                for i, img_file in enumerate(question_images)
            ]
            question_image_objects = await asyncio.gather(*question_image_tasks)

        answer_image_tasks = [
            read_and_process_image(img_file, f"{request_id}_ans{i}")
            for i, img_file in enumerate(answer_images)
        ]
        answer_image_objects = await asyncio.gather(*answer_image_tasks)

        if not answer_image_objects:
            raise HTTPException(status_code=400, detail="No valid answer images found")

        logger.info(f"[{request_id}] Processed {len(answer_image_objects)} answer images")

        all_extracted_text = ""
        # Phase 1: Extract text using LLM-Lite
        async with llm_pool.get_llm_lite() as llm_lite:
            logger.info(f"[{request_id}] Acquired LLM-Lite instance for text extraction")

            extraction_tasks = [
                extract_text_from_image(llm_lite, image, i, request_id)
                for i, image in enumerate(answer_image_objects)
            ]
            extracted_text_parts = await asyncio.gather(*extraction_tasks)
            all_extracted_text = "\n\n".join(extracted_text_parts)

            logger.info(f"[{request_id}] Text extraction completed for all pages")

        homework_result = None
        # Phase 2: Evaluate using standard LLM (gemini-2.5-flash)
        async with llm_pool.get_llm() as llm_eval:
            logger.info(f"[{request_id}] Acquired LLM instance for batch evaluation")

            homework_result = await evaluate_homework_batch(
                llm_eval,
                normalized_questions,
                question_image_objects,
                all_extracted_text,
                request_id
            )

            logger.info(f"[{request_id}] Batch evaluation completed successfully")

        # Return structured response
        response_data = {
            "status": "success",
            "request_id": request_id,
            "result": homework_result.model_dump() if isinstance(homework_result, HomeworkEvaluationResult) else homework_result,
            "full_extracted_text": all_extracted_text
        }

        logger.info(f"[{request_id}] Homework batch request completed successfully")
        return response_data

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[{request_id}] Internal server error in homework batch evaluation: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.post("/autoscore/")
async def autoscore(
    question: str = Form(..., description="Question text"),
    question_images: list[UploadFile] | None = File(default=None, description="Optional question images (1-2 images)"),
    answer_images: list[UploadFile] = File(..., description="Student answer images"),
    

):
    """Main endpoint for student answer evaluation with proper request isolation."""
    # Generate unique request ID for tracking and isolation
    request_id = str(uuid.uuid4())[:8]
    logger.info(f"[{request_id}] Starting autoscore request")
  
    try:
        # Read files concurrently and convert to PIL Images
        question_image_objects_tasks = [
            read_and_process_image(img_file, f"{request_id}_q{i}")
            for i, img_file in enumerate(question_images or [])
        ]
        answer_image_objects_tasks = [
            read_and_process_image(img_file, f"{request_id}_a{i}")
            for i, img_file in enumerate(answer_images)
        ]

        question_image_objects, answer_image_objects = await asyncio.gather(
            asyncio.gather(*question_image_objects_tasks),
            asyncio.gather(*answer_image_objects_tasks)
        )

        if not answer_image_objects:
            raise HTTPException(status_code=400, detail="No valid answer images found")

        logger.info(f"[{request_id}] Processed {len(answer_image_objects)} answer images")

        all_extracted_text = ""
        # Phase 1: OCR Text Extraction (parallel processing) using LLM-Lite
        async with llm_pool.get_llm_lite() as llm_lite:
            logger.info(f"[{request_id}] Acquired LLM-Lite instance for extraction")

            extraction_tasks = [
                extract_text_from_image(llm_lite, image, i, request_id)
                for i, image in enumerate(answer_image_objects)
            ]
            extracted_text_parts = await asyncio.gather(*extraction_tasks)
            all_extracted_text = "\n\n".join(extracted_text_parts)
            logger.info(f"[{request_id}] Text extraction completed")

        evaluation_result = None
        # Phase 2: Evaluation using standard LLM (gemini-2.5-flash)
        async with llm_pool.get_llm() as llm_eval:
            logger.info(f"[{request_id}] Acquired LLM instance for evaluation")

            evaluation_result = await evaluate_answer(
                 llm_eval, question, question_image_objects, all_extracted_text, request_id
            )
            logger.info(f"[{request_id}] Evaluation completed")

        logger.info(f"[{request_id}] Request completed successfully")

        # Return structured response
        if isinstance(evaluation_result, Result):
            return {
                "status": "success",
                "result": evaluation_result.model_dump(),
                "extracted_text": all_extracted_text,
                "request_id": request_id
            }
        else:
            logger.error(f"[{request_id}] Evaluation failed, result was not of type Result")
            raise HTTPException(status_code=500, detail="Evaluation failed")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[{request_id}] Internal server error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

async def read_and_process_image(uploaded_file: UploadFile, file_id: str) -> Image.Image:
    """Reads uploaded file content and opens it as a PIL Image."""
    if uploaded_file.content_type not in ["image/png", "image/jpeg", "image/jpg"]:
        raise HTTPException(status_code=400, detail=f"Invalid image format: {uploaded_file.content_type}")

    logger.debug(f"[{file_id}] Processing uploaded file: {uploaded_file.filename}")
    file_content = await uploaded_file.read()
    image = Image.open(BytesIO(file_content))
    logger.debug(f"[{file_id}] Image loaded: {image.size}")
    return image
