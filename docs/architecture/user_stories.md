# VR4VET Chatbot Player — User Stories

## 1. Chatbot as a Guide in VR

### 1.1 Simple Explanation of VR Controls
**As a first-time VR user**  
- I want the chatbot (NPC) to explain how to use the controllers to grab objects or navigate  
- so that I can easily understand basic VR interaction before starting the task.

**Acceptance Criteria (examples):**
- The chatbot is automatically triggered the first time a user logs in (or starts the module).
- The chatbot explains at least: how to point, how to press buttons/triggers, how to teleport (if applicable).
- The user can interrupt or skip the explanation if desired.

### 1.2 Guidance While Performing Tasks
**As a user**  
- I want the chatbot to give me real-time instructions while I complete a task  
- so that I understand the steps I must follow and do not get stuck.

**Acceptance Criteria (examples):**
- The chatbot can send a message or “speak” (audio/text) when the user reaches a certain step in the task.
- The instructions are short and precise, with an option to ask for more details.
- The chatbot can detect if the user hasn’t performed an expected step within a certain timeframe, then offer an extra hint.

### 1.3 Handling Questions From the User
**As a user**  
- I want to be able to ask the chatbot what the next step is or what I am doing wrong  
- so that I can get specific help without searching through a menu.

**Acceptance Criteria (examples):**
- The chatbot understands and answers simple questions (for example, “What should I do now?”).
- The chatbot repeats the last step the user should complete or gives a hint about the next step.
- Audio or text input (if available) must be handled so that the chatbot achieves at least 80% accuracy in understanding.

---

## 2. Memory in the Same Session

### 2.1 Short-Term Memory for User Dialogue
**As a user**  
- I want the chatbot to remember the last topics/questions in the same session  
- so that the conversation feels continuous and I don’t have to repeat myself.

**Acceptance Criteria (examples):**
- The chatbot stores (in memory) the last X interactions.
- The chatbot can refer to its previous answer (“As I mentioned earlier, you can find the fish in the laboratory...”).
- The memory is cleared upon logout/end of session, or when the user restarts.

### 2.2 Recognize the User’s Previous Actions in the Session
**As a user**  
- I want the chatbot to know which tasks I have already completed in the same session  
- so that it does not repeat unnecessary instructions.

**Acceptance Criteria (examples):**
- The system updates a “task log” when the user finishes a task.
- The chatbot checks this log before suggesting the next step.
- If a task has already been completed, the chatbot says, “You’ve already done this; would you like to move on to the next one?”

---

## 3. Proactive Help Based on Context

### 3.1 Detect User Errors and Provide Extra Tips
**As a user**  
- I want the chatbot to notice if I perform an action incorrectly multiple times  
- so that it automatically suggests an alternative approach or offers a solution.

**Acceptance Criteria (examples):**
- The system logs the number of errors (or failed attempts) in a specific task.
- After exceeding a threshold (e.g., 3 attempts), the chatbot displays a message: “It seems you’re having trouble; here’s a tip…”.
- The chatbot can differentiate between minor mistakes (e.g., pressing the wrong button once) and systematic errors (repeated incorrect actions).

### 3.2 Recognize Inactivity and Give a Nudge in the Right Direction
**As a user**  
- I want the chatbot to detect if I’m standing still or not doing anything in the game  
- so that I receive a reminder about what I can do or how to proceed.

**Acceptance Criteria (examples):**
- If the user is inactive for longer than X seconds/minutes, a message or guidance is triggered.
- The chatbot must suggest at least one concrete action: “Try opening the cabinet on the left” or “Fetch the tool from the box.”
- The inactivity flag resets once the user moves or completes an action.

---

## 4. Logging and Database

### 4.1 Logging User Actions
**As a system/database administrator**  
- I want the system to log relevant events (e.g., task started, task completed, errors, time spent)  
- so that we can analyze them later.

**Acceptance Criteria (examples):**
- Every time a task is started/completed, a record is written with a timestamp in the database.
- If the user makes an error, a new log entry is created with the error type and time.
- The log is tied to an anonymized session ID if a full user ID is not used (due to privacy).

### 4.2 Generate User Report
**As a user**  
- I want to access a simple report of my total progress and time spent  
- so that I can analyze my performance.

**Acceptance Criteria (examples):**
- The report is generated either automatically at the end of a session or on demand via a button/menu.
- The report contains completed tasks, number of errors, number of repetitions, and total time spent.
- The report can be saved as a PDF or emailed to the instructor and/or user (if approved).

---

## 5. Adaptation for Users With Special Needs

### 5.1 Special Instructions Based on Needs
**As a user with cognitive challenges**  
- I want the chatbot to offer simpler language, shorter sentences, and repeated key instructions  
- so that I am not overwhelmed and understand what to do.

**Acceptance Criteria (examples):**
- During session creation, the user (or instructor) can select a “Simplified Mode.”
- In simplified mode, the chatbot uses simpler sentence structures and repeats each step.
- The chatbot avoids complex language and technical terms (or briefly explains them if necessary).

### 5.2 Teleportation and Navigation Assistance
**As a user with mobility impairments or who gets easily disoriented in VR**  
- I want the chatbot to offer automatic teleportation to relevant areas  
- so that I don’t have to manually navigate between different parts of the scene.

**Acceptance Criteria (examples):**
- The chatbot recognizes that the user has activated a “mobility assistance” mode.
- When the user asks “Where is the laboratory?”, the chatbot suggests “Would you like to be teleported there?”.
- Teleportation requires a simple confirmation (“Yes/No”) to avoid unintended relocation.

---

## 6. RAG (Retrieval-Augmented Generation) and Microservice

### 6.1 Uploading Manuals and Documents to RAG
**As a developer/administrator**  
- I want to be able to upload relevant documentation (e.g., user manuals, learning materials) to the RAG system  
- so that the chatbot has up-to-date context to answer from.

**Acceptance Criteria (examples):**
- Ability to upload PDF/text files through a simple web interface or script.
- The documents are indexed and available to the chatbot for the next query.
- Confirmation that the document was successfully uploaded and indexed.

### 6.2 Looking Up RAG During Dialogue
**As a user**  
- I want the chatbot to provide precise information drawn from manuals or documentation  
- so that I don’t have to search external documents myself.

**Acceptance Criteria (examples):**
- When the user asks a factual question (e.g., “How much feed should the fish get?”), the chatbot searches the RAG documents and responds based on existing data.
- If the topic is not found in the documents, the chatbot replies “I’m sorry, I don’t have information on that.”
- The chatbot may link to the relevant chapter/page in the manual if appropriate.

### 6.3 Python Microservice for the Chatbot/LLM
**As a developer**  
- I want to be able to call a dedicated Python backend to handle chatbot requests  
- so that the Unity client in VR does not need large libraries or locally run models.

**Acceptance Criteria (examples):**
- The Unity project sends text-based requests to a REST endpoint (e.g., `/api/chat`).
- The Python backend returns a response within X seconds for acceptable VR flow.
- Authentication or session tokens are handled to ensure the correct access level.
