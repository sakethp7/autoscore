# # fast multi threading version includes images - Auto loads from data2.json
# # pip install google-genai streamlit pandas langsmith pymupdf pillow
# import streamlit as st
# import base64
# import os
# from typing import List, Optional, Dict, Any
# from pydantic import BaseModel, Field
# from google import genai
# from google.genai import types
# from langsmith import traceable, trace
# from langsmith.run_helpers import get_current_run_tree
# import PyPDF2
# from io import BytesIO
# import json
# from dotenv import load_dotenv
# import traceback
# import concurrent.futures
# import threading
# from concurrent.futures import ThreadPoolExecutor, as_completed
# import time
# import re

# # Load environment variables
# load_dotenv()

# # Initialize Gemini client
# client = None

# def initialize_llm(api_key: str):
#     """Initialize or update the LLM with API key"""
#     global client
#     try:
#         # Initialize Gemini client
#         client = genai.Client(api_key=api_key)
#         return True
#     except Exception as e:
#         st.error(f"Failed to initialize LLM: {str(e)}")
#         return False

# # ============= Pydantic Models for Structured Output =============

# class QuestionItem(BaseModel):
#     """Single question item"""
#     question_number: str = Field(description="Question number")
#     question_text: str = Field(description="Question text")
#     marks: float = Field(description="Maximum marks for this question")
#     has_diagram: bool = Field(description="Whether this question has a diagram")
#     diagram_sequence: int = Field(description="Sequence number of the diagram image", default=0)

# class ExtractedContent(BaseModel):
#     """Extracted content from student answer sheet"""
#     roll_number: str = Field(description="Student's roll number")
#     page_number: int = Field(description="Page number of the answer sheet")
#     content: str = Field(description="Full content in LaTeX format")
#     questions_found: List[str] = Field(description="List of question numbers found on this page")

# class QuestionEvaluation(BaseModel):
#     """Evaluation result for a single question"""
#     question_number: str = Field(description="Question number (e.g., '1', '2a', '3(i)')")
#     max_marks: float = Field(description="Maximum marks for this question")
#     total_score: float = Field(description="Total score obtained for this question")
#     error_type: str = Field(description="Type of error: conceptual_error, calculation_error, logical_error, or no_error")
#     mistakes_made: str = Field(description="Specific mistakes made in the question")
#     concepts_used: List[str] = Field(description="List of concepts that should be used to solve this question")
#     gap_analysis: str = Field(description="Specific concepts or approaches the student doesn't know")

# class EvaluationResult(BaseModel):
#     """Complete evaluation result for a student"""
#     roll_number: str = Field(description="Student's roll number")
#     questions: List[QuestionEvaluation] = Field(description="Evaluation for each question")
#     total_marks_obtained: float = Field(description="Total marks obtained")
#     total_max_marks: float = Field(description="Total maximum marks")
#     overall_percentage: float = Field(description="Overall percentage score")
#     strengths: List[str] = Field(description="Student's strengths identified")
#     areas_for_improvement: List[str] = Field(description="Areas where student needs improvement")

# # ============= Enhanced Extraction Prompt =============

# def get_extraction_prompt() -> str:
#     return """
# Extract ALL text from this image with extreme precision for student answer evaluation.

# **EXTRACTION RULES:**

# 1. **ROLL NUMBER & PAGE:** 
#    - Extract roll number from anywhere visible on the page
#    - Extract page number if visible
#    - Format: "Roll: [number], Page: [number]"

# 2. **QUESTION STRUCTURE:**
#    - Main questions: "1)", "2)", "Q1", "Question 1", etc.
#    - Sub-questions: "(i)", "(ii)", "(iii)", "(a)", "(b)", "(c)", "Part A", "Part B", etc.
#    - Nested sub-questions: "1(a)(i)", "2(b)(ii)", etc.

# 3. **MATHEMATICAL CONTENT:**
#    - Use LaTeX notation for ALL mathematical expressions
#    - Inline math: $expression$
#    - Display math: $$expression$$
#    - Preserve ALL steps, calculations, and working
#    - Include crossed-out work (mark as [CROSSED OUT: ...])

# 4. **TEXT CONTENT:**
#    - Preserve ALL written explanations
#    - Include margin notes, corrections, and annotations
#    - Mark unclear text as [UNCLEAR: best guess]

# 5. **VISUAL ELEMENTS:**
#    - Describe diagrams: [DIAGRAM: description]
#    - Describe graphs: [GRAPH: axes labels, curves, points]
#    - Describe tables: [TABLE: structure and content]
#    - Describe geometric figures: [FIGURE: shape, labels, measurements]

# 6. **FORMATTING:**
#    - Maintain original structure and indentation
#    - Preserve bullet points, numbering, and lists
#    - Keep line breaks where significant

# **OUTPUT FORMAT:**
# Roll: [number], Page: [number]

# [Question Number])
# [Original question text if visible]
# [Student's Solution:]
# [All work, steps, calculations in exact order]
# [Final answer if marked]

# [Continue for all questions on page...]

# **START EXTRACTION NOW:**"""

# # ============= Enhanced Evaluation Prompt =============

# def get_evaluation_prompt(student_answers: str, question_paper: str) -> str:
#     return f"""
# You are an expert examiner. Evaluate the student's answers based on the question paper provided.

# **QUESTION PAPER:**
# {question_paper}

# **STUDENT'S ANSWERS:**
# {student_answers}

# **EVALUATION GUIDELINES:**

# 1. **SCORING METHODOLOGY:**
#    - Conceptual Understanding: 50% of total marks
#    - Problem-solving Procedure: 20% of total marks  
#    - Final Answer Accuracy: 10% of total marks
#    - Mathematical Formulas/Methods: 20% of total marks
   
# 2. **ERROR CLASSIFICATION:**
#    - conceptual_error: Wrong concept or theory applied
#    - calculation_error: Arithmetic or computational mistakes
#    - logical_error: Flawed reasoning or incorrect sequence
#    - no_error: Completely correct answer

# 3. **EVALUATION REQUIREMENTS:**
#    - Extract question numbers and their maximum marks from the question paper
#    - Identify which questions the student attempted
#    - Award partial marks based on correct steps even if final answer is wrong
#    - Provide specific feedback on mistakes made
#    - Identify concepts the student used correctly and incorrectly
#    - Suggest specific areas for improvement

# 4. **OUTPUT REQUIREMENTS:**
#    - For each question found, provide: question_number, max_marks, total_score, error_type, mistakes_made, concepts_used, gap_analysis
#    - Calculate total marks and percentage
#    - Identify strengths and improvement areas
#    - Be fair but accurate in scoring

# Evaluate all questions found in the student's answers against the question paper."""

# # ============= Helper Functions =============

# def pdf_to_images_base64(pdf_file) -> List[tuple[int, str]]:
#     """Convert PDF pages to base64 encoded images using PyMuPDF"""
#     try:
#         import fitz  # PyMuPDF
#         import io
#         from PIL import Image
        
#         pdf_bytes = pdf_file.read()
#         pdf_document = fitz.open(stream=pdf_bytes, filetype="pdf")
#         pages_base64 = []
        
#         for page_num in range(pdf_document.page_count):
#             page = pdf_document[page_num]
#             # Convert page to image (pixmap) with higher resolution
#             pix = page.get_pixmap(matrix=fitz.Matrix(2.0, 2.0))  # 2x scaling for better quality
#             # Convert pixmap to PIL Image
#             img_data = pix.tobytes("png")
#             image = Image.open(io.BytesIO(img_data))
            
#             # Convert to base64
#             buffered = io.BytesIO()
#             image.save(buffered, format="JPEG", quality=95)
#             img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
#             pages_base64.append((page_num + 1, img_base64))
        
#         pdf_document.close()
#         return pages_base64
        
#     except ImportError:
#         st.error("PyMuPDF not installed. Please install it:")
#         st.code("pip install pymupdf")
#         st.info("PyMuPDF is a pure Python library - no additional system dependencies required!")
#         return []
#     except Exception as e:
#         st.error(f"Error processing PDF: {str(e)}")
#         return []

# def image_to_base64(image_file) -> str:
#     """Convert uploaded image to base64"""
#     image_file.seek(0)
#     return base64.b64encode(image_file.read()).decode('utf-8')

# def load_questions_from_json_file(json_filepath: str) -> tuple[List[Dict], List[str]]:
#     """Load questions from JSON file and automatically load images from same directory"""
#     try:
#         import os
        
#         with open(json_filepath, 'r') as file:
#             json_content = json.load(file)
        
#         questions = []
#         question_images = []
#         image_sequence = 1
        
#         # Get directory of JSON file for image loading
#         json_dir = os.path.dirname(json_filepath)
        
#         for item in json_content:
#             question = {
#                 'question_number': str(item['question_number']),
#                 'question_text': item['question'],
#                 'marks': float(item['marks']),
#                 'has_diagram': item.get('has_diagram', False),
#                 'diagram_sequence': 0
#             }
            
#             # Handle diagram
#             if question['has_diagram'] and item.get('image_path'):
#                 image_filename = item['image_path']
#                 image_filepath = os.path.join(json_dir, image_filename)
                
#                 # Check if image file exists
#                 if os.path.exists(image_filepath):
#                     try:
#                         with open(image_filepath, 'rb') as img_file:
#                             img_base64 = base64.b64encode(img_file.read()).decode('utf-8')
#                             question_images.append(img_base64)
#                             question['diagram_sequence'] = image_sequence
#                             image_sequence += 1
#                     except Exception as e:
#                         st.warning(f"Could not load image {image_filename}: {str(e)}")
#                         question['has_diagram'] = False
#                 else:
#                     st.warning(f"Image {image_filename} not found at {image_filepath}")
#                     question['has_diagram'] = False
            
#             questions.append(question)
        
#         return questions, question_images
        
#     except FileNotFoundError:
#         st.error(f"data2.json file not found at {json_filepath}")
#         return [], []
#     except Exception as e:
#         st.error(f"Error loading questions from {json_filepath}: {str(e)}")
#         return [], []

# def auto_load_questions():
#     """Automatically load questions from data2.json on app startup"""
#     json_filepath = "data2.json"
    
#     if os.path.exists(json_filepath):
#         questions, images = load_questions_from_json_file(json_filepath)
        
#         if questions:
#             st.session_state.questions = questions
#             st.session_state.question_images = images
#             return True, len(questions), len(images)
#         else:
#             return False, 0, 0
#     else:
#         st.error(f"data2.json file not found in current directory")
#         return False, 0, 0

# @traceable(name="extract_content_from_page")
# def extract_content_from_page_thread(page_data: tuple) -> Dict:
#     """Extract content from a single page using LLM - thread-safe version"""
#     page_num, base64_img = page_data
    
#     if not client:
#         return {
#             'page_number': page_num,
#             'content': f"Error: LLM not initialized for page {page_num}",
#             'success': False
#         }
    
#     try:
#         # Create content with text and image
#         contents = [
#             types.Content(
#                 role="user",
#                 parts=[
#                     types.Part.from_text(text=get_extraction_prompt()),
#                     types.Part.from_bytes(
#                         mime_type="image/jpeg",
#                         data=base64.b64decode(base64_img)
#                     ),
#                 ],
#             ),
#         ]
        
#         # Generate response
#         response = client.models.generate_content(
#             model="gemini-2.5-flash",
#             contents=contents,
#             config=types.GenerateContentConfig()
#         )
        
#         return {
#             'page_number': page_num,
#             'content': response.text,
#             'success': True
#         }
        
#     except Exception as e:
#         error_msg = f"Error extracting content from page {page_num}: {str(e)}"
#         return {
#             'page_number': page_num,
#             'content': error_msg,
#             'success': False
#         }

# def extract_content_parallel(pages_data: List[tuple], max_workers: int = 3) -> List[Dict]:
#     """Extract content from multiple pages in parallel"""
#     results = []
    
#     with ThreadPoolExecutor(max_workers=max_workers) as executor:
#         # Submit all tasks
#         future_to_page = {executor.submit(extract_content_from_page_thread, page_data): page_data[0] 
#                          for page_data in pages_data}
        
#         # Collect results as they complete
#         for future in as_completed(future_to_page):
#             page_num = future_to_page[future]
#             try:
#                 result = future.result()
#                 results.append(result)
#             except Exception as e:
#                 st.error(f"Page {page_num} failed: {str(e)}")
#                 results.append({
#                     'page_number': page_num,
#                     'content': f"Failed to process page {page_num}: {str(e)}",
#                     'success': False
#                 })
    
#     # Sort results by page number
#     results.sort(key=lambda x: x['page_number'])
#     return results

# @traceable(name="evaluate_all_answers")
# def evaluate_all_answers(student_answers: str, question_paper: str, question_images: List[str] = None) -> EvaluationResult:
#     """Evaluate all answers using structured LLM output"""
    
#     if not client:
#         return EvaluationResult(
#             roll_number="Error",
#             questions=[],
#             total_marks_obtained=0.0,
#             total_max_marks=0.0,
#             overall_percentage=0.0,
#             strengths=["LLM not initialized"],
#             areas_for_improvement=["API configuration needed"]
#         )
    
#     try:
#         # Create evaluation prompt with JSON format request
#         evaluation_prompt = get_evaluation_prompt(student_answers, question_paper) + """

# Please provide your evaluation in the following JSON format:
# {
#     "roll_number": "student roll number",
#     "questions": [
#         {
#             "question_number": "1",
#             "max_marks": 10.0,
#             "total_score": 8.5,
#             "error_type": "calculation_error",
#             "mistakes_made": "Arithmetic error in step 3",
#             "concepts_used": ["Quadratic equations", "Factorization"],
#             "gap_analysis": "Student needs to practice careful arithmetic"
#         }
#     ],
#     "total_marks_obtained": 85.5,
#     "total_max_marks": 100.0,
#     "overall_percentage": 85.5,
#     "strengths": ["Good conceptual understanding", "Clear presentation"],
#     "areas_for_improvement": ["Arithmetic accuracy", "Time management"]
# }"""
        
#         # Build parts list
#         parts = [types.Part.from_text(text=evaluation_prompt)]
        
#         # Add question images if available
#         if question_images:
#             for img_base64 in question_images:
#                 parts.append(
#                     types.Part.from_bytes(
#                         mime_type="image/jpeg",
#                         data=base64.b64decode(img_base64)
#                     )
#                 )
        
#         # Create content
#         contents = [
#             types.Content(
#                 role="user",
#                 parts=parts,
#             ),
#         ]
        
#         # Generate response
#         response = client.models.generate_content(
#             model="gemini-2.0-flash",
#             contents=contents,
#             config=types.GenerateContentConfig()
#         )
        
#         # Extract JSON from response
#         response_text = response.text
        
#         # Try to find JSON in the response
#         import re
#         json_match = re.search(r'\{[\s\S]*\}', response_text)
#         if json_match:
#             json_str = json_match.group()
#             result_dict = json.loads(json_str)
#             return EvaluationResult(**result_dict)
#         else:
#             # Fallback: create a basic result
#             return EvaluationResult(
#                 roll_number="Unknown",
#                 questions=[],
#                 total_marks_obtained=0.0,
#                 total_max_marks=0.0,
#                 overall_percentage=0.0,
#                 strengths=["Could not parse evaluation"],
#                 areas_for_improvement=["Please try again"]
#             )
        
#     except Exception as e:
#         st.error(f"Error during evaluation: {str(e)}")
#         return EvaluationResult(
#             roll_number="Error",
#             questions=[],
#             total_marks_obtained=0.0,
#             total_max_marks=0.0,
#             overall_percentage=0.0,
#             strengths=[],
#             areas_for_improvement=[f"Evaluation failed: {str(e)}"]
#         )

# def build_question_paper_string(questions: List[Dict]) -> str:
#     """Build question paper string from stored questions"""
#     question_paper = ""
    
#     for i, q in enumerate(questions):
#         question_paper += f"{q['question_number']}. {q['question_text']}"
        
#         if q['has_diagram']:
#             question_paper += f" (has diagram - image {q['diagram_sequence']})"
        
#         question_paper += f" ({q['marks']} marks)\n\n"
    
#     return question_paper

# # ============= Streamlit UI =============

# def main():
#     st.set_page_config(page_title="Enhanced Multipage Autoscore", layout="wide")
    
#     st.title("📚 Enhanced Multipage Autoscore")
#     st.markdown("### AI-Powered Automated Student Answer Evaluation")
    
#     # Initialize session state
#     if 'questions' not in st.session_state:
#         st.session_state.questions = []
#     if 'question_images' not in st.session_state:
#         st.session_state.question_images = []
#     if 'extracted_content' not in st.session_state:
#         st.session_state.extracted_content = []
#     if 'evaluation_results' not in st.session_state:
#         st.session_state.evaluation_results = None
#     if 'phase1_complete' not in st.session_state:
#         st.session_state.phase1_complete = False
#     if 'questions_loaded' not in st.session_state:
#         st.session_state.questions_loaded = False
    
#     # Auto-load questions from data2.json on first run
#     if not st.session_state.questions_loaded:
#         success, num_questions, num_images = auto_load_questions()
#         if success:
#             st.success(f"✅ Auto-loaded {num_questions} questions from data2.json")
#             if num_images > 0:
#                 st.info(f"📊 Loaded {num_images} question diagrams")
#             st.session_state.questions_loaded = True
#         else:
#             st.error("❌ Failed to auto-load questions from data2.json")
#         st.rerun()
    
#     # Sidebar for configuration
#     with st.sidebar:
#         st.header("⚙️ Configuration")
        
#         api_key = st.text_input("Gemini API Key", type="password", value=os.getenv("GEMINI_API_KEY", ""))
#         if api_key:
#             os.environ["GEMINI_API_KEY"] = api_key
#             if initialize_llm(api_key):
#                 st.success("✅ API Key configured")
#         else:
#             st.warning("⚠️ Please enter your Gemini API Key")
        
#         st.divider()
        
#         st.subheader("🔧 Processing Settings")
#         max_workers = st.selectbox("Parallel Extraction Threads", [1, 2, 3, 4, 5], index=2)
#         st.info(f"Will process {max_workers} pages simultaneously")
        
#         st.divider()
        
#         st.subheader("📊 Progress Tracker")
#         if st.session_state.questions:
#             st.success("✅ Questions Loaded from data2.json")
#         else:
#             st.info("⏳ Loading Questions...")
            
#         if st.session_state.phase1_complete:
#             st.success("✅ Phase 1: Content Extracted")
#         else:
#             st.info("⏳ Phase 1: Extract Answers")
            
#         if st.session_state.evaluation_results:
#             st.success("✅ Phase 2: Evaluation Complete")
#         else:
#             st.info("⏳ Phase 2: Evaluate Answers")
        
#         st.divider()
        
#         if st.session_state.questions:
#             st.subheader("📝 Loaded Questions")
#             st.info(f"Total Questions: {len(st.session_state.questions)}")
#             total_marks = sum(q['marks'] for q in st.session_state.questions)
#             st.info(f"Total Marks: {total_marks}")
            
#             diagrams_count = sum(1 for q in st.session_state.questions if q['has_diagram'])
#             if diagrams_count > 0:
#                 st.info(f"Questions with Diagrams: {diagrams_count}")
                
#             # Show question list
#             with st.expander("📋 Question List", expanded=False):
#                 for q in st.session_state.questions:
#                     diagram_indicator = " 📊" if q['has_diagram'] else ""
#                     st.text(f"Q{q['question_number']}: {q['marks']} marks{diagram_indicator}")
        
#         # Refresh questions button
#         st.divider()
#         if st.button("🔄 Reload Questions from data2.json"):
#             success, num_questions, num_images = auto_load_questions()
#             if success:
#                 st.success(f"✅ Reloaded {num_questions} questions")
#                 if num_images > 0:
#                     st.info(f"📊 Loaded {num_images} diagrams")
#                 st.rerun()
    
#     # Show question paper preview if questions are loaded
#     if st.session_state.questions:
#         with st.expander("📝 Question Paper Preview", expanded=False):
#             question_paper = build_question_paper_string(st.session_state.questions)
#             st.markdown(question_paper)
    
#     # Main content area with tabs (removed Phase 0)
#     tab1, tab2, tab3 = st.tabs(["📤 Phase 1: Extract", "📊 Phase 2: Evaluate", "📈 Results"])
    
#     # Tab 1: Upload and Extract (with Multi-threading)
#     with tab1:
#         st.header("Phase 1: Upload and Extract Student Answers")
#         st.markdown("🚀 **Multi-threaded Processing**: Pages will be processed in parallel for faster extraction")
        
#         if not st.session_state.questions:
#             st.warning("⚠️ No questions loaded. Please ensure data2.json is in the current directory and click 'Reload Questions'")
#             return
        
#         col1, col2 = st.columns([1, 1])
        
#         with col1:
#             st.subheader("Upload Answer Sheets")
            
#             file_type = st.radio("Select file type:", ["Images", "PDF"])
            
#             if file_type == "PDF":
#                 uploaded_file = st.file_uploader(
#                     "Choose PDF file",
#                     type=['pdf'],
#                     help="Upload student answer sheet as PDF"
#                 )
#             else:
#                 uploaded_files = st.file_uploader(
#                     "Choose image files",
#                     type=['png', 'jpg', 'jpeg'],
#                     accept_multiple_files=True,
#                     help="Upload student answer sheets as images"
#                 )
            
#             if st.button("🔍 Extract Content (Multi-threaded)", type="primary"):
#                 if not api_key or not client:
#                     st.error("Please provide and configure Gemini API Key in the sidebar")
#                 elif file_type == "PDF" and uploaded_file:
#                     with st.spinner("Converting PDF to images and extracting content in parallel..."):
#                         pages = pdf_to_images_base64(uploaded_file)
#                         if pages:
#                             st.info(f"📄 Processing {len(pages)} pages with {max_workers} parallel threads")
                            
#                             # Create progress tracking
#                             progress_bar = st.progress(0)
#                             status_text = st.empty()
                            
#                             # Process pages in parallel
#                             start_time = time.time()
#                             results = extract_content_parallel(pages, max_workers)
#                             end_time = time.time()
                            
#                             # Update progress
#                             progress_bar.progress(1.0)
#                             processing_time = end_time - start_time
#                             status_text.success(f"✅ Completed in {processing_time:.2f} seconds")
                            
#                             # Convert results to expected format
#                             extracted = []
#                             for result in results:
#                                 extracted.append({
#                                     'page_number': result['page_number'],
#                                     'content': result['content']
#                                 })
                            
#                             st.session_state.extracted_content = extracted
#                             st.session_state.phase1_complete = True
                            
#                             successful_pages = sum(1 for r in results if r['success'])
#                             st.success(f"✅ Successfully extracted content from {successful_pages}/{len(pages)} pages")
                            
#                             if successful_pages < len(pages):
#                                 st.warning(f"⚠️ {len(pages) - successful_pages} pages had errors")
                            
#                             st.info("📊 Phase 2 is ready to start!")
                        
#                 elif file_type == "Images" and uploaded_files:
#                     with st.spinner("Extracting content from images in parallel..."):
#                         # Prepare image data
#                         pages_data = []
#                         for i, img_file in enumerate(uploaded_files):
#                             try:
#                                 img_file.seek(0)
#                                 base64_img = image_to_base64(img_file)
#                                 pages_data.append((i + 1, base64_img))
#                             except Exception as e:
#                                 st.error(f"Failed to process {img_file.name}: {str(e)}")
#                                 continue
                        
#                         if pages_data:
#                             st.info(f"📄 Processing {len(pages_data)} images with {max_workers} parallel threads")
                            
#                             # Create progress tracking
#                             progress_bar = st.progress(0)
#                             status_text = st.empty()
                            
#                             # Process images in parallel
#                             start_time = time.time()
#                             results = extract_content_parallel(pages_data, max_workers)
#                             end_time = time.time()
                            
#                             # Update progress
#                             progress_bar.progress(1.0)
#                             processing_time = end_time - start_time
#                             status_text.success(f"✅ Completed in {processing_time:.2f} seconds")
                            
#                             # Convert results to expected format
#                             extracted = []
#                             for result in results:
#                                 extracted.append({
#                                     'page_number': result['page_number'],
#                                     'content': result['content']
#                                 })
                            
#                             st.session_state.extracted_content = extracted
#                             if extracted:
#                                 st.session_state.phase1_complete = True
                                
#                                 successful_pages = sum(1 for r in results if r['success'])
#                                 st.success(f"✅ Successfully extracted content from {successful_pages}/{len(results)} images")
                                
#                                 if successful_pages < len(results):
#                                     st.warning(f"⚠️ {len(results) - successful_pages} images had errors")
                                
#                                 st.info("📊 Phase 2 is ready to start!")
#                 else:
#                     st.warning("Please upload files first")
        
#         with col2:
#             st.subheader("Extracted Content Preview")
            
#             if st.session_state.extracted_content:
#                 # Display content with LaTeX rendering
#                 st.markdown("**Extracted Content:**")
                
#                 # Use expander for each page
#                 for page_data in st.session_state.extracted_content:
#                     with st.expander(f"Page {page_data['page_number']}", expanded=(len(st.session_state.extracted_content) == 1)):
#                         # Display content with proper LaTeX formatting
#                         content_display = page_data['content']
#                         content_display = content_display.replace("$$", "\n$$\n")
#                         st.markdown(content_display)
#             else:
#                 st.info("No content extracted yet. Upload files and click 'Extract Content'")
    
#     # Tab 2: Evaluation
#     with tab2:
#         st.header("Phase 2: Automated Answer Evaluation")
        
#         if not st.session_state.questions:
#             st.warning("⚠️ Questions not loaded. Please ensure data2.json is available and click 'Reload Questions'")
#         elif not st.session_state.extracted_content:
#             st.warning("⚠️ Please complete Phase 1 first - extract student answers")
#         else:
#             st.info("🤖 **Automated Evaluation**: The LLM will evaluate answers based on the questions from data2.json.")
            
#             # Show question paper preview
#             question_paper = build_question_paper_string(st.session_state.questions)
            
#             with st.expander("📝 Preview: Question Paper", expanded=False):
#                 st.markdown(question_paper)
#                 if st.session_state.question_images:
#                     st.info(f"📊 {len(st.session_state.question_images)} diagram(s) will be included in evaluation")
            
#             # Combine all extracted content
#             full_content = "\n".join([f"<page {p['page_number']}>\n{p['content']}\n</page {p['page_number']}>" 
#                                      for p in st.session_state.extracted_content])
            
#             with st.expander("📄 Preview: Combined Student Answers", expanded=False):
#                 st.markdown(full_content)
            
#             st.divider()
            
#             # Evaluation button
#             col1, col2, col3 = st.columns([1, 2, 1])
#             with col2:
#                 if st.button("🎯 Start Automated Evaluation", type="primary", use_container_width=True):
#                     with st.spinner("🤖 AI is evaluating all answers with question diagrams... This may take a few moments."):
#                         try:
#                             # Perform evaluation with question images
#                             results = evaluate_all_answers(
#                                 full_content, 
#                                 question_paper, 
#                                 st.session_state.question_images if st.session_state.question_images else None
#                             )
                            
#                             # Store results
#                             st.session_state.evaluation_results = results
                            
#                             st.success("✅ Automated evaluation completed! Check the Results tab.")
#                             st.balloons()
                            
#                         except Exception as e:
#                             st.error(f"❌ Evaluation failed: {str(e)}")
#                             st.error("Please check your API key and try again.")
    
#     # Tab 3: Results
#     with tab3:
#         st.header("📊 Evaluation Results")
        
#         if st.session_state.evaluation_results:
#             results = st.session_state.evaluation_results
            
#             # Summary metrics
#             col1, col2, col3, col4 = st.columns(4)
            
#             with col1:
#                 st.metric("Roll Number", results.roll_number)
#             with col2:
#                 st.metric("Total Score", f"{results.total_marks_obtained:.1f}/{results.total_max_marks:.0f}")
#             with col3:
#                 st.metric("Percentage", f"{results.overall_percentage:.1f}%")
#             with col4:
#                 # Grade calculation
#                 percentage = results.overall_percentage
#                 if percentage >= 90:
#                     grade = "A+"
#                 elif percentage >= 80:
#                     grade = "A"
#                 elif percentage >= 70:
#                     grade = "B"
#                 elif percentage >= 60:
#                     grade = "C"
#                 elif percentage >= 50:
#                     grade = "D"
#                 else:
#                     grade = "F"
#                 st.metric("Grade", grade)
            
#             st.divider()
            
#             # Detailed results table
#             st.subheader("Question-wise Performance")
            
#             # Create a dataframe for better visualization
#             import pandas as pd
            
#             data = []
#             for q_eval in results.questions:
#                 data.append({
#                     "Question": q_eval.question_number,
#                     "Max Marks": q_eval.max_marks,
#                     "Score Obtained": f"{q_eval.total_score:.1f}",
#                     "Percentage": f"{(q_eval.total_score/q_eval.max_marks*100):.1f}%",
#                     "Error Type": q_eval.error_type.replace('_', ' ').title()
#                 })
            
#             df = pd.DataFrame(data)
#             st.dataframe(df, use_container_width=True, hide_index=True)
            
#             st.divider()
            
#             # Detailed analysis for each question
#             st.subheader("Detailed Analysis")
            
#             for q_eval in results.questions:
#                 percentage = (q_eval.total_score/q_eval.max_marks*100) if q_eval.max_marks > 0 else 0
                
#                 with st.expander(f"Question {q_eval.question_number} - {q_eval.total_score:.1f}/{q_eval.max_marks} ({percentage:.1f}%)"):
#                     col_a, col_b = st.columns(2)
                    
#                     with col_a:
#                         st.markdown("**Performance Summary:**")
#                         st.metric("Score", f"{q_eval.total_score:.1f}/{q_eval.max_marks}")
                        
#                         if q_eval.error_type != "no_error":
#                             st.error(f"**Error Type:** {q_eval.error_type.replace('_', ' ').title()}")
#                         else:
#                             st.success("**Perfect Answer!** ✅")
                        
#                         st.markdown("**Concepts Involved:**")
#                         if q_eval.concepts_used:
#                             for concept in q_eval.concepts_used[:5]:
#                                 st.markdown(f"- {concept}")
#                         else:
#                             st.markdown("- Analysis pending")
                    
#                     with col_b:
#                         st.markdown("**Mistakes Made:**")
#                         if q_eval.mistakes_made:
#                             st.info(q_eval.mistakes_made)
#                         else:
#                             st.success("No major mistakes found!")
                        
#                         st.markdown("**Gap Analysis:**")
#                         if q_eval.gap_analysis:
#                             st.warning(q_eval.gap_analysis)
#                         else:
#                             st.info("Student shows good understanding.")
            
#             st.divider()
            
#             # Overall feedback
#             col1, col2 = st.columns(2)
            
#             with col1:
#                 st.subheader("💪 Strengths")
#                 if results.strengths:
#                     for strength in results.strengths:
#                         st.success(f"✓ {strength}")
#                 else:
#                     st.info("Strengths analysis will appear here.")
            
#             with col2:
#                 st.subheader("📚 Areas for Improvement")
#                 if results.areas_for_improvement:
#                     for area in results.areas_for_improvement:
#                         st.warning(f"→ {area}")
#                 else:
#                     st.info("Keep up the excellent work!")
            
#             # Export results
#             st.divider()
#             st.subheader("Export Results")
            
#             col1, col2, col3 = st.columns(3)
            
#             with col1:
#                 # Convert to JSON for download
#                 results_dict = results.dict()
#                 json_str = json.dumps(results_dict, indent=2)
#                 st.download_button(
#                     label="📥 Download JSON",
#                     data=json_str,
#                     file_name=f"evaluation_{results.roll_number}.json",
#                     mime="application/json"
#                 )
            
#             with col2:
#                 # Create CSV summary
#                 csv_data = "Question,Max Marks,Score Obtained,Percentage,Error Type\n"
#                 for q in results.questions:
#                     percentage = (q.total_score/q.max_marks*100) if q.max_marks > 0 else 0
#                     csv_data += f"{q.question_number},{q.max_marks},{q.total_score},{percentage:.1f}%,{q.error_type}\n"
                
#                 # Add summary row
#                 csv_data += f"TOTAL,{results.total_max_marks},{results.total_marks_obtained},{results.overall_percentage:.1f}%,\n"
                
#                 st.download_button(
#                     label="📥 Download CSV",
#                     data=csv_data,
#                     file_name=f"scores_{results.roll_number}.csv",
#                     mime="text/csv"
#                 )
            
#             with col3:
#                 # Generate detailed report
#                 report = f"""
# AUTOMATED STUDENT EVALUATION REPORT
# ==================================
# Roll Number: {results.roll_number}
# Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}
# Evaluated by: AI Assistant

# OVERALL PERFORMANCE
# ------------------
# Total Marks: {results.total_marks_obtained:.1f}/{results.total_max_marks:.0f}
# Percentage: {results.overall_percentage:.1f}%
# Grade: {grade}

# QUESTION-WISE SCORES
# -------------------
# """
#                 for q in results.questions:
#                     percentage = (q.total_score/q.max_marks*100) if q.max_marks > 0 else 0
#                     report += f"\nQuestion {q.question_number}: {q.total_score:.1f}/{q.max_marks} ({percentage:.1f}%)\n"
#                     if q.error_type != "no_error":
#                         report += f"  - Error Type: {q.error_type}\n"
#                     if q.mistakes_made:
#                         report += f"  - Issues: {q.mistakes_made[:100]}...\n"
                
#                 report += "\n\nSTRENGTHS:\n"
#                 for s in results.strengths:
#                     report += f"- {s}\n"
                
#                 report += "\nAREAS FOR IMPROVEMENT:\n"
#                 for a in results.areas_for_improvement:
#                     report += f"- {a}\n"
                
#                 st.download_button(
#                     label="📄 Download Report",
#                     data=report,
#                     file_name=f"report_{results.roll_number}.txt",
#                     mime="text/plain"
#                 )
        
#         else:
#             st.info("No evaluation results yet. Complete all previous phases first.")

# if __name__ == "__main__":
#     main()
# fast multi threading version includes images - Auto loads from data2.json
# pip install google-genai streamlit pandas langsmith pymupdf pillow
import streamlit as st
import base64
import os
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from google import genai
from google.genai import types
from langsmith import traceable, trace
from langsmith.run_helpers import get_current_run_tree
import PyPDF2
from io import BytesIO
import json
from dotenv import load_dotenv
import traceback
import concurrent.futures
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import re

# Load environment variables
load_dotenv()

# Initialize Gemini client
client = None

def initialize_llm(api_key: str):
    """Initialize or update the LLM with API key"""
    global client
    try:
        # Initialize Gemini client
        client = genai.Client(api_key=api_key)
        return True
    except Exception as e:
        st.error(f"Failed to initialize LLM: {str(e)}")
        return False

# ============= Pydantic Models for Structured Output =============

class QuestionItem(BaseModel):
    """Single question item"""
    question_number: str = Field(description="Question number")
    question_text: str = Field(description="Question text")
    marks: float = Field(description="Maximum marks for this question")
    has_diagram: bool = Field(description="Whether this question has a diagram")
    diagram_sequence: int = Field(description="Sequence number of the diagram image", default=0)

class ExtractedContent(BaseModel):
    """Extracted content from student answer sheet"""
    roll_number: str = Field(description="Student's roll number")
    page_number: int = Field(description="Page number of the answer sheet")
    content: str = Field(description="Full content in LaTeX format")
    questions_found: List[str] = Field(description="List of question numbers found on this page")

class QuestionEvaluation(BaseModel):
    """Evaluation result for a single question"""
    question_number: str = Field(description="Question number (e.g., '1', '2a', '3(i)')")
    max_marks: float = Field(description="Maximum marks for this question")
    total_score: float = Field(description="Total score obtained for this question")
    error_type: str = Field(description="Type of error: conceptual_error, calculation_error, logical_error, or no_error")
    mistakes_made: str = Field(description="Specific mistakes made in the question,what student wrote and what should have been written")
    concepts_used: List[str] = Field(description="List of concepts that should be used to solve this question")
    gap_analysis: str = Field(description="Specific concepts or approaches the student doesn't know")

class EvaluationResult(BaseModel):
    """Complete evaluation result for a student"""
    roll_number: str = Field(description="Student's roll number")
    questions: List[QuestionEvaluation] = Field(description="Evaluation for each question")
    total_marks_obtained: float = Field(description="Total marks obtained")
    total_max_marks: float = Field(description="Total maximum marks")
    overall_percentage: float = Field(description="Overall percentage score")
    strengths: List[str] = Field(description="Student's strengths identified")
    areas_for_improvement: List[str] = Field(description="Areas where student needs improvement")

# ============= Enhanced Extraction Prompt =============

def get_extraction_prompt() -> str:
    return """
Extract ALL text from this image with extreme precision for student answer evaluation.

**EXTRACTION RULES:**

1. **ROLL NUMBER & PAGE:** 
   - Extract roll number from anywhere visible on the page
   - Extract page number if visible
   - Format: "Roll: [number], Page: [number]"

2. **QUESTION STRUCTURE:**
   - Main questions: "1)", "2)", "Q1", "Question 1", etc.
   - Sub-questions: "(i)", "(ii)", "(iii)", "(a)", "(b)", "(c)", "Part A", "Part B", etc.
   - Nested sub-questions: "1(a)(i)", "2(b)(ii)", etc.

3. **MATHEMATICAL CONTENT:**
   - Use LaTeX notation for ALL mathematical expressions
   - Inline math: $expression$
   - Display math: $$expression$$
   - Preserve ALL steps, calculations, and working
   - Include crossed-out work (mark as [CROSSED OUT: ...])

4. **TEXT CONTENT:**
   - Preserve ALL written explanations
   - Include margin notes, corrections, and annotations
   - Mark unclear text as [UNCLEAR: best guess]

5. **VISUAL ELEMENTS:**
   - Describe diagrams: [DIAGRAM: description]
   - Describe graphs: [GRAPH: axes labels, curves, points]
   - Describe tables: [TABLE: structure and content]
   - Describe geometric figures: [FIGURE: shape, labels, measurements]

6. **FORMATTING:**
   - Maintain original structure and indentation
   - Preserve bullet points, numbering, and lists
   - Keep line breaks where significant

**OUTPUT FORMAT:**
Roll: [number], Page: [number]

[Question Number])
[Original question text if visible]
[Student's Solution:]
[All work, steps, calculations in exact order]
[Final answer if marked]

[Continue for all questions on page...]

**START EXTRACTION NOW:**"""

# ============= Enhanced Evaluation Prompt =============

def get_evaluation_prompt(student_answers: str, question_paper: str) -> str:
    return f"""
You are an expert examiner. Evaluate the student's answers based on the question paper provided.

**QUESTION PAPER:**
{question_paper}

**STUDENT'S ANSWERS:**
{student_answers}

**EVALUATION GUIDELINES:**

1. **SCORING METHODOLOGY:**
   - Conceptual Understanding: 50% of total marks
   - Problem-solving Procedure: 20% of total marks  
   - Final Answer Accuracy: 10% of total marks
   - Mathematical Formulas/Methods: 20% of total marks
   
2. **ERROR CLASSIFICATION:**
   - conceptual_error: Wrong concept or theory applied
   - calculation_error: Arithmetic or computational mistakes
   - logical_error: Flawed reasoning or incorrect sequence
   - no_error: Completely correct answer

3. **EVALUATION REQUIREMENTS:**
   - Extract question numbers and their maximum marks from the question paper
   - Identify which questions the student attempted
   - Award partial marks based on correct steps even if final answer is wrong
   - Provide specific feedback on mistakes made,specify what student wrote and what should have been written
   - Identify concepts the student used correctly and incorrectly
   - Suggest specific areas for improvement

4. **OUTPUT REQUIREMENTS:**
   - For each question found, provide: question_number, max_marks, total_score, error_type, mistakes_made, concepts_used, gap_analysis
   - Calculate total marks and percentage
   - Identify strengths and improvement areas
   - Be fair but accurate in scoring

Evaluate all questions found in the student's answers against the question paper."""

# ============= Helper Functions =============

def pdf_to_images_base64(pdf_file) -> List[tuple[int, str]]:
    """Convert PDF pages to base64 encoded images using PyMuPDF"""
    try:
        import fitz  # PyMuPDF
        import io
        from PIL import Image
        
        pdf_bytes = pdf_file.read()
        pdf_document = fitz.open(stream=pdf_bytes, filetype="pdf")
        pages_base64 = []
        
        for page_num in range(pdf_document.page_count):
            page = pdf_document[page_num]
            # Convert page to image (pixmap) with higher resolution
            pix = page.get_pixmap(matrix=fitz.Matrix(2.0, 2.0))  # 2x scaling for better quality
            # Convert pixmap to PIL Image
            img_data = pix.tobytes("png")
            image = Image.open(io.BytesIO(img_data))
            
            # Convert to base64
            buffered = io.BytesIO()
            image.save(buffered, format="JPEG", quality=95)
            img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
            pages_base64.append((page_num + 1, img_base64))
        
        pdf_document.close()
        return pages_base64
        
    except ImportError:
        st.error("PyMuPDF not installed. Please install it:")
        st.code("pip install pymupdf")
        st.info("PyMuPDF is a pure Python library - no additional system dependencies required!")
        return []
    except Exception as e:
        st.error(f"Error processing PDF: {str(e)}")
        return []

def image_to_base64(image_file) -> str:
    """Convert uploaded image to base64"""
    image_file.seek(0)
    return base64.b64encode(image_file.read()).decode('utf-8')

def load_questions_from_json_file(json_filepath: str) -> tuple[List[Dict], List[str]]:
    """Load questions from JSON file and automatically load images from same directory"""
    try:
        import os
        
        with open(json_filepath, 'r') as file:
            json_content = json.load(file)
        
        questions = []
        question_images = []
        image_sequence = 1
        
        # Get directory of JSON file for image loading
        json_dir = os.path.dirname(json_filepath)
        
        for item in json_content:
            question = {
                'question_number': str(item['question_number']),
                'question_text': item['question'],
                'marks': float(item['marks']),
                'has_diagram': item.get('has_diagram', False),
                'diagram_sequence': 0
            }
            
            # Handle diagram
            if question['has_diagram'] and item.get('image_path'):
                image_filename = item['image_path']
                image_filepath = os.path.join(json_dir, image_filename)
                
                # Check if image file exists
                if os.path.exists(image_filepath):
                    try:
                        with open(image_filepath, 'rb') as img_file:
                            img_base64 = base64.b64encode(img_file.read()).decode('utf-8')
                            question_images.append(img_base64)
                            question['diagram_sequence'] = image_sequence
                            image_sequence += 1
                    except Exception as e:
                        st.warning(f"Could not load image {image_filename}: {str(e)}")
                        question['has_diagram'] = False
                else:
                    st.warning(f"Image {image_filename} not found at {image_filepath}")
                    question['has_diagram'] = False
            
            questions.append(question)
        
        return questions, question_images
        
    except FileNotFoundError:
        st.error(f"data2.json file not found at {json_filepath}")
        return [], []
    except Exception as e:
        st.error(f"Error loading questions from {json_filepath}: {str(e)}")
        return [], []

def auto_load_questions():
    """Automatically load questions from data2.json on app startup"""
    json_filepath = "data2.json"
    
    if os.path.exists(json_filepath):
        questions, images = load_questions_from_json_file(json_filepath)
        
        if questions:
            st.session_state.questions = questions
            st.session_state.question_images = images
            return True, len(questions), len(images)
        else:
            return False, 0, 0
    else:
        st.error(f"data2.json file not found in current directory")
        return False, 0, 0

@traceable(name="extract_content_from_page")
def extract_content_from_page_thread(page_data: tuple) -> Dict:
    """Extract content from a single page using LLM - thread-safe version"""
    page_num, base64_img = page_data
    
    if not client:
        return {
            'page_number': page_num,
            'content': f"Error: LLM not initialized for page {page_num}",
            'success': False
        }
    
    try:
        # Create content with text and image
        contents = [
            types.Content(
                role="user",
                parts=[
                    types.Part.from_text(text=get_extraction_prompt()),
                    types.Part.from_bytes(
                        mime_type="image/jpeg",
                        data=base64.b64decode(base64_img)
                    ),
                ],
            ),
        ]
        
        # Generate response
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=contents,
            config=types.GenerateContentConfig()
        )
        
        return {
            'page_number': page_num,
            'content': response.text,
            'success': True
        }
        
    except Exception as e:
        error_msg = f"Error extracting content from page {page_num}: {str(e)}"
        return {
            'page_number': page_num,
            'content': error_msg,
            'success': False
        }

def extract_content_parallel(pages_data: List[tuple], max_workers: int = 3) -> List[Dict]:
    """Extract content from multiple pages in parallel"""
    results = []
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_page = {executor.submit(extract_content_from_page_thread, page_data): page_data[0] 
                         for page_data in pages_data}
        
        # Collect results as they complete
        for future in as_completed(future_to_page):
            page_num = future_to_page[future]
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                st.error(f"Page {page_num} failed: {str(e)}")
                results.append({
                    'page_number': page_num,
                    'content': f"Failed to process page {page_num}: {str(e)}",
                    'success': False
                })
    
    # Sort results by page number
    results.sort(key=lambda x: x['page_number'])
    return results

@traceable(name="evaluate_all_answers")
def evaluate_all_answers(student_answers: str, question_paper: str, question_images: List[str] = None) -> EvaluationResult:
    """Evaluate all answers using structured LLM output"""
    
    if not client:
        return EvaluationResult(
            roll_number="Error",
            questions=[],
            total_marks_obtained=0.0,
            total_max_marks=0.0,
            overall_percentage=0.0,
            strengths=["LLM not initialized"],
            areas_for_improvement=["API configuration needed"]
        )
    
    try:
        # Create evaluation prompt with JSON format request
        evaluation_prompt = get_evaluation_prompt(student_answers, question_paper) + """

Please provide your evaluation in the following JSON format:
{
    "roll_number": "student roll number",
    "questions": [
        {
            "question_number": "1",
            "max_marks": 10.0,
            "total_score": 8.5,
            "error_type": "calculation_error",
            "mistakes_made": "Arithmetic error in step 3",
            "concepts_used": ["Quadratic equations", "Factorization"],
            "gap_analysis": "Student needs to practice careful arithmetic"
        }
    ],
    "total_marks_obtained": 85.5,
    "total_max_marks": 100.0,
    "overall_percentage": 85.5,
    "strengths": ["Good conceptual understanding", "Clear presentation"],
    "areas_for_improvement": ["Arithmetic accuracy", "Time management"]
}"""
        
        # Build parts list
        parts = [types.Part.from_text(text=evaluation_prompt)]
        
        # Add question images if available
        if question_images:
            for img_base64 in question_images:
                parts.append(
                    types.Part.from_bytes(
                        mime_type="image/jpeg",
                        data=base64.b64decode(img_base64)
                    )
                )
        
        # Create content
        contents = [
            types.Content(
                role="user",
                parts=parts,
            ),
        ]
        
        # Generate response
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=contents,
            config=types.GenerateContentConfig()
        )
        
        # Extract JSON from response
        response_text = response.text
        
        # Try to find JSON in the response
        import re
        json_match = re.search(r'\{[\s\S]*\}', response_text)
        if json_match:
            json_str = json_match.group()
            result_dict = json.loads(json_str)
            return EvaluationResult(**result_dict)
        else:
            # Fallback: create a basic result
            return EvaluationResult(
                roll_number="Unknown",
                questions=[],
                total_marks_obtained=0.0,
                total_max_marks=0.0,
                overall_percentage=0.0,
                strengths=["Could not parse evaluation"],
                areas_for_improvement=["Please try again"]
            )
        
    except Exception as e:
        st.error(f"Error during evaluation: {str(e)}")
        return EvaluationResult(
            roll_number="Error",
            questions=[],
            total_marks_obtained=0.0,
            total_max_marks=0.0,
            overall_percentage=0.0,
            strengths=[],
            areas_for_improvement=[f"Evaluation failed: {str(e)}"]
        )

def build_question_paper_string(questions: List[Dict]) -> str:
    """Build question paper string from stored questions"""
    question_paper = ""
    
    for i, q in enumerate(questions):
        question_paper += f"{q['question_number']}. {q['question_text']}"
        
        if q['has_diagram']:
            question_paper += f" (has diagram - image {q['diagram_sequence']})"
        
        question_paper += f" ({q['marks']} marks)\n\n"
    
    return question_paper

# ============= Streamlit UI =============

def main():
    st.set_page_config(page_title="Enhanced Multipage Autoscore", layout="wide")
    
    st.title("📚 Enhanced Multipage Autoscore")
    st.markdown("### AI-Powered Automated Student Answer Evaluation")
    
    # Initialize session state
    if 'questions' not in st.session_state:
        st.session_state.questions = []
    if 'question_images' not in st.session_state:
        st.session_state.question_images = []
    if 'extracted_content' not in st.session_state:
        st.session_state.extracted_content = []
    if 'evaluation_results' not in st.session_state:
        st.session_state.evaluation_results = None
    if 'phase1_complete' not in st.session_state:
        st.session_state.phase1_complete = False
    if 'questions_loaded' not in st.session_state:
        st.session_state.questions_loaded = False
    
    # Auto-load questions from data2.json on first run
    if not st.session_state.questions_loaded:
        success, num_questions, num_images = auto_load_questions()
        if success:
            st.success(f"✅ Auto-loaded {num_questions} questions from data2.json")
            if num_images > 0:
                st.info(f"📊 Loaded {num_images} question diagrams")
            st.session_state.questions_loaded = True
        else:
            st.error("❌ Failed to auto-load questions from data2.json")
        st.rerun()
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        api_key = st.text_input("Gemini API Key", type="password", value=os.getenv("GEMINI_API_KEY", ""))
        if api_key:
            os.environ["GEMINI_API_KEY"] = api_key
            if initialize_llm(api_key):
                st.success("✅ API Key configured")
        else:
            st.warning("⚠️ Please enter your Gemini API Key")
        
        st.divider()
        
        st.subheader("🔧 Processing Settings")
        max_workers = st.selectbox("Parallel Extraction Threads", [1, 2, 3, 4, 5], index=2)
        st.info(f"Will process {max_workers} pages simultaneously")
        
        st.divider()
        
        st.subheader("📊 Progress Tracker")
        if st.session_state.questions:
            st.success("✅ Questions Configured")
        else:
            st.info("⏳ Configure Questions in Phase 1")
            
        if st.session_state.phase1_complete:
            st.success("✅ Phase 1: Content Extracted")
        else:
            st.info("⏳ Phase 1: Configure & Extract")
            
        if st.session_state.evaluation_results:
            st.success("✅ Phase 2: Evaluation Complete")
        else:
            st.info("⏳ Phase 2: Evaluate Answers")
        
        st.divider()
        
        if st.session_state.questions:
            st.subheader("📝 Current Questions")
            st.info(f"Total Questions: {len(st.session_state.questions)}")
            total_marks = sum(q['marks'] for q in st.session_state.questions)
            st.info(f"Total Marks: {total_marks}")
            
            diagrams_count = sum(1 for q in st.session_state.questions if q['has_diagram'])
            if diagrams_count > 0:
                st.info(f"Questions with Diagrams: {diagrams_count}")
                
            # Show question list
            with st.expander("📋 Question List", expanded=False):
                for q in st.session_state.questions:
                    diagram_indicator = " 📊" if q['has_diagram'] else ""
                    st.text(f"Q{q['question_number']}: {q['marks']} marks{diagram_indicator}")
        
        # Refresh questions button
        st.divider()
        if st.button("🔄 Reload Default Questions from data2.json"):
            success, num_questions, num_images = auto_load_questions()
            if success:
                st.success(f"✅ Reloaded {num_questions} questions")
                if num_images > 0:
                    st.info(f"📊 Loaded {num_images} diagrams")
                st.rerun()
            else:
                st.error("❌ Could not load from data2.json")
    
    # Show question paper preview if questions are loaded
    if st.session_state.questions:
        with st.expander("📝 Question Paper Preview", expanded=False):
            question_paper = build_question_paper_string(st.session_state.questions)
            st.markdown(question_paper)
    
    # Main content area with tabs
    tab1, tab2, tab3 = st.tabs(["📝 Phase 1: Configure & Extract", "📊 Phase 2: Evaluate", "📈 Results"])
    
    # Tab 1: Upload and Extract (with Multi-threading)
    with tab1:
        st.header("Phase 1: Configure Questions & Extract Student Answers")
        st.markdown("🚀 **Multi-threaded Processing**: Pages will be processed in parallel for faster extraction")
        
        # Question source selection
        st.subheader("📝 Step 1: Choose Question Source")
        question_source = st.radio(
            "Select question source:",
            ["📋 Use Existing Questions (data2.json)", "📤 Upload Custom Questions"],
            horizontal=True
        )
        
        if question_source == "📋 Use Existing Questions (data2.json)":
            # Show existing questions in editable text area
            col_q1, col_q2 = st.columns([2, 1])
            
            with col_q1:
                st.markdown("**Edit Questions (JSON format):**")
                
                # Convert current questions back to JSON for editing
                if st.session_state.questions:
                    questions_json = []
                    for q in st.session_state.questions:
                        questions_json.append({
                            "question_number": int(q['question_number']) if q['question_number'].isdigit() else q['question_number'],
                            "question": q['question_text'],
                            "has_diagram": q['has_diagram'],
                            "image_path": f"{q['question_number']}.jpeg" if q['has_diagram'] else None,
                            "marks": q['marks']
                        })
                    default_json = json.dumps(questions_json, indent=2)
                else:
                    # Default questions from data2.json format
                    default_json = '''[
  {
    "question_number": 1,
    "question": "$ \\\\text{If } P(A) = \\\\frac{7}{13},\\\\ P(B) = \\\\frac{9}{13} \\\\text{ and } P(A \\\\cap B) = \\\\frac{4}{13},\\\\ \\\\text{evaluate } P(A \\\\mid B). $",
    "has_diagram": false,
    "image_path": null,
    "marks": 2
  },
  {
    "question_number": 2,
    "question": "A family has two children. What is the probability that both the children are boys given that at least one of them is a boy?",
    "has_diagram": false,
    "image_path": null,
    "marks": 2
  },
  {
    "question_number": 3,
    "question": "Ten cards numbered 1 to 10 are placed in a box, mixed thoroughly and one card is drawn randomly. If it is known that the number on the drawn card is more than 3, what is the probability that it is an even number?",
    "has_diagram": false,
    "image_path": null,
    "marks": 2
  }
]'''
                
                edited_questions = st.text_area(
                    "Questions JSON:",
                    value=default_json,
                    height=400,
                    help="Edit the questions in JSON format. You can modify question text, marks, or add/remove questions."
                )
                
                # Optional image upload for questions with diagrams
                st.markdown("**Upload Question Images (Optional):**")
                question_images_files = st.file_uploader(
                    "Upload images for questions that have diagrams",
                    type=['png', 'jpg', 'jpeg'],
                    accept_multiple_files=True,
                    help="Upload images for questions where has_diagram is true. Name them as question_number.jpeg (e.g., 5.jpeg)"
                )
                
                if st.button("🔄 Update Questions", type="secondary"):
                    try:
                        # Parse the edited JSON
                        questions_data = json.loads(edited_questions)
                        
                        # Process uploaded images
                        uploaded_images = {}
                        if question_images_files:
                            for img_file in question_images_files:
                                img_file.seek(0)
                                uploaded_images[img_file.name] = base64.b64encode(img_file.read()).decode('utf-8')
                        
                        # Convert to internal format
                        questions = []
                        question_images = []
                        image_sequence = 1
                        
                        for item in questions_data:
                            question = {
                                'question_number': str(item['question_number']),
                                'question_text': item['question'],
                                'marks': float(item['marks']),
                                'has_diagram': item.get('has_diagram', False),
                                'diagram_sequence': 0
                            }
                            
                            # Handle diagram
                            if question['has_diagram'] and item.get('image_path'):
                                image_filename = item['image_path']
                                if image_filename in uploaded_images:
                                    question_images.append(uploaded_images[image_filename])
                                    question['diagram_sequence'] = image_sequence
                                    image_sequence += 1
                                else:
                                    st.warning(f"Image {image_filename} not found for question {item['question_number']}")
                                    question['has_diagram'] = False
                            
                            questions.append(question)
                        
                        # Update session state
                        st.session_state.questions = questions
                        st.session_state.question_images = question_images
                        
                        st.success(f"✅ Updated {len(questions)} questions successfully!")
                        if question_images:
                            st.info(f"📊 Loaded {len(question_images)} question images")
                        st.rerun()
                        
                    except json.JSONDecodeError as e:
                        st.error(f"❌ Invalid JSON format: {str(e)}")
                    except Exception as e:
                        st.error(f"❌ Error updating questions: {str(e)}")
            
            with col_q2:
                st.markdown("**Current Questions Summary:**")
                if st.session_state.questions:
                    for q in st.session_state.questions:
                        diagram_indicator = " 📊" if q['has_diagram'] else ""
                        st.text(f"Q{q['question_number']}: {q['marks']} marks{diagram_indicator}")
                    
                    total_marks = sum(q['marks'] for q in st.session_state.questions)
                    st.info(f"Total: {len(st.session_state.questions)} questions, {total_marks} marks")
                else:
                    st.info("No questions loaded yet")
        
        else:  # Upload Custom Questions
            col_u1, col_u2 = st.columns([2, 1])
            
            with col_u1:
                st.markdown("**Upload Questions JSON File:**")
                json_file = st.file_uploader(
                    "Choose JSON file",
                    type=['json'],
                    help="Upload JSON file with question structure"
                )
                
                st.markdown("**Upload Question Images:**")
                question_images_files = st.file_uploader(
                    "Choose image files for questions",
                    type=['png', 'jpg', 'jpeg'],
                    accept_multiple_files=True,
                    help="Upload all images referenced in the JSON file",
                    key="custom_images"
                )
                
                if st.button("📤 Load Custom Questions", type="secondary"):
                    if json_file:
                        try:
                            # Load questions from uploaded JSON
                            json_content = json.load(json_file)
                            
                            # Process uploaded images
                            uploaded_images = {}
                            if question_images_files:
                                for img_file in question_images_files:
                                    img_file.seek(0)
                                    uploaded_images[img_file.name] = base64.b64encode(img_file.read()).decode('utf-8')
                            
                            # Convert to internal format
                            questions = []
                            question_images = []
                            image_sequence = 1
                            
                            for item in json_content:
                                question = {
                                    'question_number': str(item['question_number']),
                                    'question_text': item['question'],
                                    'marks': float(item['marks']),
                                    'has_diagram': item.get('has_diagram', False),
                                    'diagram_sequence': 0
                                }
                                
                                # Handle diagram
                                if question['has_diagram'] and item.get('image_path'):
                                    image_filename = item['image_path']
                                    if image_filename in uploaded_images:
                                        question_images.append(uploaded_images[image_filename])
                                        question['diagram_sequence'] = image_sequence
                                        image_sequence += 1
                                    else:
                                        st.warning(f"Image {image_filename} not found for question {item['question_number']}")
                                        question['has_diagram'] = False
                                
                                questions.append(question)
                            
                            # Update session state
                            st.session_state.questions = questions
                            st.session_state.question_images = question_images
                            
                            st.success(f"✅ Loaded {len(questions)} custom questions!")
                            if question_images:
                                st.info(f"📊 Loaded {len(question_images)} question images")
                            st.rerun()
                            
                        except Exception as e:
                            st.error(f"❌ Error loading custom questions: {str(e)}")
                    else:
                        st.warning("Please upload a JSON file first")
            
            with col_u2:
                st.markdown("**Expected JSON format:**")
                sample_json = [
                    {
                        "question_number": 1,
                        "question": "Solve the equation x² + 5x + 6 = 0",
                        "has_diagram": False,
                        "image_path": None,
                        "marks": 10
                    },
                    {
                        "question_number": 2,
                        "question": "Find the area of triangle ABC",
                        "has_diagram": True,
                        "image_path": "triangle.jpg",
                        "marks": 8
                    }
                ]
                st.json(sample_json)
        
        st.divider()
        
        # Answer sheet processing section
        st.subheader("📤 Step 2: Upload and Extract Student Answers")
        
        if not st.session_state.questions:
            st.warning("⚠️ Please configure questions first in Step 1")
            return
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("Upload Answer Sheets")
            
            file_type = st.radio("Select answer sheet file type:", ["Images", "PDF"])
            
            if file_type == "PDF":
                uploaded_file = st.file_uploader(
                    "Choose PDF file",
                    type=['pdf'],
                    help="Upload student answer sheet as PDF",
                    key="answer_pdf"
                )
            else:
                uploaded_files = st.file_uploader(
                    "Choose image files",
                    type=['png', 'jpg', 'jpeg'],
                    accept_multiple_files=True,
                    help="Upload student answer sheets as images",
                    key="answer_images"
                )
            
            if st.button("🔍 Extract Content (Multi-threaded)", type="primary"):
                if not api_key or not client:
                    st.error("Please provide and configure Gemini API Key in the sidebar")
                elif file_type == "PDF" and uploaded_file:
                    with st.spinner("Converting PDF to images and extracting content in parallel..."):
                        pages = pdf_to_images_base64(uploaded_file)
                        if pages:
                            st.info(f"📄 Processing {len(pages)} pages with {max_workers} parallel threads")
                            
                            # Create progress tracking
                            progress_bar = st.progress(0)
                            status_text = st.empty()
                            
                            # Process pages in parallel
                            start_time = time.time()
                            results = extract_content_parallel(pages, max_workers)
                            end_time = time.time()
                            
                            # Update progress
                            progress_bar.progress(1.0)
                            processing_time = end_time - start_time
                            status_text.success(f"✅ Completed in {processing_time:.2f} seconds")
                            
                            # Convert results to expected format
                            extracted = []
                            for result in results:
                                extracted.append({
                                    'page_number': result['page_number'],
                                    'content': result['content']
                                })
                            
                            st.session_state.extracted_content = extracted
                            st.session_state.phase1_complete = True
                            
                            successful_pages = sum(1 for r in results if r['success'])
                            st.success(f"✅ Successfully extracted content from {successful_pages}/{len(pages)} pages")
                            
                            if successful_pages < len(pages):
                                st.warning(f"⚠️ {len(pages) - successful_pages} pages had errors")
                            
                            st.info("📊 Phase 2 is ready to start!")
                        
                elif file_type == "Images" and uploaded_files:
                    with st.spinner("Extracting content from images in parallel..."):
                        # Prepare image data
                        pages_data = []
                        for i, img_file in enumerate(uploaded_files):
                            try:
                                img_file.seek(0)
                                base64_img = image_to_base64(img_file)
                                pages_data.append((i + 1, base64_img))
                            except Exception as e:
                                st.error(f"Failed to process {img_file.name}: {str(e)}")
                                continue
                        
                        if pages_data:
                            st.info(f"📄 Processing {len(pages_data)} images with {max_workers} parallel threads")
                            
                            # Create progress tracking
                            progress_bar = st.progress(0)
                            status_text = st.empty()
                            
                            # Process images in parallel
                            start_time = time.time()
                            results = extract_content_parallel(pages_data, max_workers)
                            end_time = time.time()
                            
                            # Update progress
                            progress_bar.progress(1.0)
                            processing_time = end_time - start_time
                            status_text.success(f"✅ Completed in {processing_time:.2f} seconds")
                            
                            # Convert results to expected format
                            extracted = []
                            for result in results:
                                extracted.append({
                                    'page_number': result['page_number'],
                                    'content': result['content']
                                })
                            
                            st.session_state.extracted_content = extracted
                            if extracted:
                                st.session_state.phase1_complete = True
                                
                                successful_pages = sum(1 for r in results if r['success'])
                                st.success(f"✅ Successfully extracted content from {successful_pages}/{len(results)} images")
                                
                                if successful_pages < len(results):
                                    st.warning(f"⚠️ {len(results) - successful_pages} images had errors")
                                
                                st.info("📊 Phase 2 is ready to start!")
                else:
                    st.warning("Please upload files first")
        
        with col2:
            st.subheader("Extracted Content Preview")
            
            if st.session_state.extracted_content:
                # Display content with LaTeX rendering
                st.markdown("**Extracted Content:**")
                
                # Use expander for each page
                for page_data in st.session_state.extracted_content:
                    with st.expander(f"Page {page_data['page_number']}", expanded=(len(st.session_state.extracted_content) == 1)):
                        # Display content with proper LaTeX formatting
                        content_display = page_data['content']
                        content_display = content_display.replace("$$", "\n$$\n")
                        st.markdown(content_display)
            else:
                st.info("No content extracted yet. Upload files and click 'Extract Content'")
    
    # Tab 2: Evaluation
    with tab2:
        st.header("Phase 2: Automated Answer Evaluation")
        
        if not st.session_state.questions:
            st.warning("⚠️ Questions not loaded. Please configure questions in Phase 1 first.")
        elif not st.session_state.extracted_content:
            st.warning("⚠️ Please complete Phase 1 first - extract student answers")
        else:
            st.info("🤖 **Automated Evaluation**: The LLM will evaluate answers based on the configured questions.")
            
            # Show question paper preview
            question_paper = build_question_paper_string(st.session_state.questions)
            
            with st.expander("📝 Preview: Question Paper", expanded=False):
                st.markdown(question_paper)
                if st.session_state.question_images:
                    st.info(f"📊 {len(st.session_state.question_images)} diagram(s) will be included in evaluation")
            
            # Combine all extracted content
            full_content = "\n".join([f"<page {p['page_number']}>\n{p['content']}\n</page {p['page_number']}>" 
                                     for p in st.session_state.extracted_content])
            
            with st.expander("📄 Preview: Combined Student Answers", expanded=False):
                st.markdown(full_content)
            
            st.divider()
            
            # Evaluation button
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                if st.button("🎯 Start Automated Evaluation", type="primary", use_container_width=True):
                    with st.spinner("🤖 AI is evaluating all answers with question diagrams... This may take a few moments."):
                        try:
                            # Perform evaluation with question images
                            results = evaluate_all_answers(
                                full_content, 
                                question_paper, 
                                st.session_state.question_images if st.session_state.question_images else None
                            )
                            
                            # Store results
                            st.session_state.evaluation_results = results
                            
                            st.success("✅ Automated evaluation completed! Check the Results tab.")
                            st.balloons()
                            
                        except Exception as e:
                            st.error(f"❌ Evaluation failed: {str(e)}")
                            st.error("Please check your API key and try again.")
    
    # Tab 3: Results
    with tab3:
        st.header("📊 Evaluation Results")
        
        if st.session_state.evaluation_results:
            results = st.session_state.evaluation_results
            
            # Summary metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Roll Number", results.roll_number)
            with col2:
                st.metric("Total Score", f"{results.total_marks_obtained:.1f}/{results.total_max_marks:.0f}")
            with col3:
                st.metric("Percentage", f"{results.overall_percentage:.1f}%")
            with col4:
                # Grade calculation
                percentage = results.overall_percentage
                if percentage >= 90:
                    grade = "A+"
                elif percentage >= 80:
                    grade = "A"
                elif percentage >= 70:
                    grade = "B"
                elif percentage >= 60:
                    grade = "C"
                elif percentage >= 50:
                    grade = "D"
                else:
                    grade = "F"
                st.metric("Grade", grade)
            
            st.divider()
            
            # Detailed results table
            st.subheader("Question-wise Performance")
            
            # Create a dataframe for better visualization
            import pandas as pd
            
            data = []
            for q_eval in results.questions:
                data.append({
                    "Question": q_eval.question_number,
                    "Max Marks": q_eval.max_marks,
                    "Score Obtained": f"{q_eval.total_score:.1f}",
                    "Percentage": f"{(q_eval.total_score/q_eval.max_marks*100):.1f}%",
                    "Error Type": q_eval.error_type.replace('_', ' ').title()
                })
            
            df = pd.DataFrame(data)
            st.dataframe(df, use_container_width=True, hide_index=True)
            
            st.divider()
            
            # Detailed analysis for each question
            st.subheader("Detailed Analysis")
            
            for q_eval in results.questions:
                percentage = (q_eval.total_score/q_eval.max_marks*100) if q_eval.max_marks > 0 else 0
                
                with st.expander(f"Question {q_eval.question_number} - {q_eval.total_score:.1f}/{q_eval.max_marks} ({percentage:.1f}%)"):
                    col_a, col_b = st.columns(2)
                    
                    with col_a:
                        st.markdown("**Performance Summary:**")
                        st.metric("Score", f"{q_eval.total_score:.1f}/{q_eval.max_marks}")
                        
                        if q_eval.error_type != "no_error":
                            st.error(f"**Error Type:** {q_eval.error_type.replace('_', ' ').title()}")
                        else:
                            st.success("**Perfect Answer!** ✅")
                        
                        st.markdown("**Concepts Involved:**")
                        if q_eval.concepts_used:
                            for concept in q_eval.concepts_used[:5]:
                                st.markdown(f"- {concept}")
                        else:
                            st.markdown("- Analysis pending")
                    
                    with col_b:
                        st.markdown("**Mistakes Made:**")
                        if q_eval.mistakes_made:
                            st.info(q_eval.mistakes_made)
                        else:
                            st.success("No major mistakes found!")
                        
                        st.markdown("**Gap Analysis:**")
                        if q_eval.gap_analysis:
                            st.warning(q_eval.gap_analysis)
                        else:
                            st.info("Student shows good understanding.")
            
            st.divider()
            
            # Overall feedback
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("💪 Strengths")
                if results.strengths:
                    for strength in results.strengths:
                        st.success(f"✓ {strength}")
                else:
                    st.info("Strengths analysis will appear here.")
            
            with col2:
                st.subheader("📚 Areas for Improvement")
                if results.areas_for_improvement:
                    for area in results.areas_for_improvement:
                        st.warning(f"→ {area}")
                else:
                    st.info("Keep up the excellent work!")
            
            # Export results
            st.divider()
            st.subheader("Export Results")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                # Convert to JSON for download
                results_dict = results.dict()
                json_str = json.dumps(results_dict, indent=2)
                st.download_button(
                    label="📥 Download JSON",
                    data=json_str,
                    file_name=f"evaluation_{results.roll_number}.json",
                    mime="application/json"
                )
            
            with col2:
                # Create CSV summary
                csv_data = "Question,Max Marks,Score Obtained,Percentage,Error Type\n"
                for q in results.questions:
                    percentage = (q.total_score/q.max_marks*100) if q.max_marks > 0 else 0
                    csv_data += f"{q.question_number},{q.max_marks},{q.total_score},{percentage:.1f}%,{q.error_type}\n"
                
                # Add summary row
                csv_data += f"TOTAL,{results.total_max_marks},{results.total_marks_obtained},{results.overall_percentage:.1f}%,\n"
                
                st.download_button(
                    label="📥 Download CSV",
                    data=csv_data,
                    file_name=f"scores_{results.roll_number}.csv",
                    mime="text/csv"
                )
            
            with col3:
                # Generate detailed report
                report = f"""
AUTOMATED STUDENT EVALUATION REPORT
==================================
Roll Number: {results.roll_number}
Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}
Evaluated by: AI Assistant

OVERALL PERFORMANCE
------------------
Total Marks: {results.total_marks_obtained:.1f}/{results.total_max_marks:.0f}
Percentage: {results.overall_percentage:.1f}%
Grade: {grade}

QUESTION-WISE SCORES
-------------------
"""
                for q in results.questions:
                    percentage = (q.total_score/q.max_marks*100) if q.max_marks > 0 else 0
                    report += f"\nQuestion {q.question_number}: {q.total_score:.1f}/{q.max_marks} ({percentage:.1f}%)\n"
                    if q.error_type != "no_error":
                        report += f"  - Error Type: {q.error_type}\n"
                    if q.mistakes_made:
                        report += f"  - Issues: {q.mistakes_made[:100]}...\n"
                
                report += "\n\nSTRENGTHS:\n"
                for s in results.strengths:
                    report += f"- {s}\n"
                
                report += "\nAREAS FOR IMPROVEMENT:\n"
                for a in results.areas_for_improvement:
                    report += f"- {a}\n"
                
                st.download_button(
                    label="📄 Download Report",
                    data=report,
                    file_name=f"report_{results.roll_number}.txt",
                    mime="text/plain"
                )
        
        else:
            st.info("No evaluation results yet. Complete all previous phases first.")

if __name__ == "__main__":
    main()