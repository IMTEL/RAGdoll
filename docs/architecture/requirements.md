
# Architectural Drivers / Architectural Significant Requirements
*This requirement document is based on a template that me (Tobias Fremming) and Sverre Nystad made in Cogito in order to make it easier to define architectural drivers in the organization, based on the curriculum in the course Software Architecture TDT4240.*

This document presents the key architectural drivers and significant requirements for the IMTEL chatbot system developed in IT2901-11 during spring 2025. It includes functional and non-functional (quality) requirements shape the system’s behavior, performance, and operational attributes in order to ensure that the needs of its end users and stakeholders are met, while maintaining a high quality standard.


## Functional Requirements
*Functional requirements specify the exact behaviors, actions, and functionalities the chatbot system must provide. They detail the operations the system must perform under various conditions and outline the services it must deliver.*

The functional requirements are divided into three different priorities: 

- **High Priority** - Essential for core system functionality; without these, the system’s ability to deliver a satisfactory experience is significantly compromised.
- **Medium Priority** - Important for system functionality but not critical; the system can still operate without them.
- **Low Priority** - Non-essential but desirable features that enhance the user experience without affecting core functionality.

The functional requirements are listed below:

### **VR chatbot interface**
| ID   | Requirement Description | Priority |
|------|-------------------------|----------|
| FR1.1| The system must allow users to ask questions at any point | High |
| FR1.2| The chatbot must provide an automatic explanation of VR controls when prompted. | High |
| FR1.3| The chatbot must explain pointing, pressing buttons, and teleporting if applicable.| Medium |
| FR1.4| The chatbot must provide real-time instructions during tasks. | High |
| FR1.5| The chatbot must detect user inactivity and offer hints. | High |
| FR1.6| The chatbot must allow users to ask about the next step or issues they face, and provide answers from the context. | High |
| FR1.7| The chatbot must understand and respond accurately to at least 80% of user queries. | High |

### **Session Memory**
| ID   | Requirement Description | Priority |
|------|-------------------------|----------|
| FR2.1| The chatbot must retain short-term memory of the last few interactions. | High |
| FR2.2| Short term memory in chat will not introduce unnecessary processing overhead making interactions faster. | High |
| FR2.3| The chatbot must refer to past interactions when relevant. | High |
| FR2.4| The system must handle queries in different languages. | Medium |
| FR2.5| The system must clear session memory upon session termination. | High |
| FR2.6| The chatbot must log completed tasks and avoid redundant instructions. | High |


### **Proactive Help Based on Context**
| ID   | Requirement Description | Priority |
|------|-------------------------|----------|
| FR3.1| The chatbot must detect repeated mistakes and offer assistance. | High |
| FR3.2| The system must recommend a learning path based on the previous user interactions in the session | Low |
| FR3.3| The chatbot must differentiate between minor and systematic errors. | Low |
| FR3.4| The chatbot must recognize user inactivity and provide guidance. | High |
| FR3.5| The chatbot must suggest specific actions based on context. | High |
| FR3.6| The actions suggested by chatbot should be relevant for the situation at least 80% of the time | High |
 
### **Logging and Database**
| ID   | Requirement Description | Priority |
|------|-------------------------|----------|
| FR4.1| The system must log relevant user actions with timestamps. | High |
| FR4.2| The system must record user mistakes and completion times. | Medium |
| FR4.3| Logs must be associated with anonymized session IDs for privacy. | High |
| FR4.4| Users must be able to generate progress reports. | Low |
| FR4.5| Reports must include completed tasks, errors, repetitions, and total session time. | Low |
| FR4.6| Reports must be downloadable as PDF or sent via email. | Low |



### **RAG (Retrieval-Augmented Generation) and Microservices**
| ID   | Requirement Description | Priority |
|------|-------------------------|----------|
| FR5.1| The system must allow uploading of manuals and documents for chatbot reference. | Medium |
| FR5.2| Documents must be indexed and searchable by the chatbot. | High |
| FR5.3| The chatbot must retrieve relevant information from uploaded documents. | High |
| FR5.4| If no relevant data is found, the chatbot must decide if the question is relevant, and answer accordingly by LLM without context | Low |
| FR5.5| The system should be able to track the performance of users in quizzes. | Low |





## Quality Attributes
*Quality attributes are the system's non-functional requirements that specify the system's operational characteristics. They define the system's behavior, performance, and other qualities that are not directly related to the system's functionality.*

### **Usability**
| ID   | Requirement Description | Priority |
|------|-------------------------|----------|
| U1 | 	The system must have an intuitive VR interface so that a new user can perform basic actions (e.g., ask questions, navigate the environment) within 30 seconds. | High |
| U2 | The system must provide clear visual or audio feedback whenever a user attempts an action (e.g., button press, speech input) to confirm the system has registered the input. | Medium |
| U3 | 	The system must comply with WCAG 2.1 Level AA guidelines for visual/auditory cues in VR to ensure accessibility for users with disabilities. | Low |
| U4 | 	The language should be understandable for any user | Low |
| U5 | The chatbot should not reply too slowly | High |
| U6 | The chatbot should provide audio and visual ques when thinking | Medium |
| U7 | The chatbot must offer a skip/interrupt option if the user does not want to listen to or read the entire explanation of VR controls. | Low |
| U8 | All important instructions (including error messages) must be presented in a concise and easily understandable manner. | Medium |


### **Performance**
| ID   | Requirement Description | Priority |
|------|-------------------------|----------|
| P1 | The chatbot must respond to user queries (including RAG-based lookups) within 7 seconds on average under normal load conditions. | High |
| P2 | The chatbot should detect user inactivity (e.g., no input or movement) within 10 seconds and proactively offer assistance. | High |
| P3 | 	When providing real-time instructions, the chatbot’s prompts or hints must appear with negligible delay (under 1 second) once a trigger event is detected in VR. | High |
| P4 | The chatbot should detect user inactivity (e.g., no input or movement) within 10 seconds and proactively offer assistance. | High |
| P5 | Streaming should be used in a way that reduces bottleneck in chatbot response generation | Low |


### **Modifiability**
| ID   | Requirement Description | Priority |
|------|-------------------------|----------|
| M1 | The system must be designed to allow for easy modification and extension of features without significant rework or refactoring. | High |
| M2 | The system must be able to change COTS (Commercial Off-The-Shelf) components with only local changes.| High |
| M3 | The system must be able to get new functionalities without much refactoring of existant code.| High |
| M4 | The system must be able to easily change database without any side effects | High |
| M5 | The chatbot microservice must be modular so that updates to the LLM or RAG indexing can be made without requiring changes to the VR front-end. | High |
| M6 | The system’s configuration (e.g., inactivity time threshold, repeated-mistake threshold) must be easily adjustable via environment variables or config files without code changes. | High |

### **Interoperability**
| ID   | Requirement Description | Priority |
|------|-------------------------|----------|
| I1 | The VR client (Unity or another engine) must communicate with the Python chatbot backend via a documented REST API (e.g., POST /api/chat) for all conversation-related interactions. | High |
| I2 | The RAG service must implement an open interface (e.g., an API endpoint or library) for uploading and indexing new training manuals or documents used by the chatbot. | Low |



### **Deployment**
| ID   | Requirement Description | Priority |
|------|-------------------------|----------|
| D1 | The system must enable straightforward deployment processes for updates and new features that introduce minimal downtime or defects. | High |
| D2 | The system must support automated deployment to streamline the release process. | Low |

### **Availability**
| ID   | Requirement Description | Priority |
|------|-------------------------|----------|
| A1 | System uptime must be 95% | Medium |
| A2 | The system must be able to recover from failures within 15 minutes. | Low |
| A3 | The system must have redundant failover mechanisms to ensure continuity during outages. | Low |

### **Testability**
| ID   | Requirement Description | Priority |
|------|-------------------------|----------|
| T1 | The system must be designed to allow efficient testing of new features and updates to ensure functionality without extensive manual intervention. | High |
| T2 | The system must have a test suite that covers all essential features | High |
| T3 | The system must be able to mock external services for testing | Medium |


### **Security**
| ID   | Requirement Description | Priority |
|------|-------------------------|----------|
| S1 | The system must be secure to protect the user's data and personal information. | High |
| S3 | The system must delete all user generated data at the termination of a session in case of a security breach | low |
| S4 | The system must contain login or unique session ID that cannot be replicated in order to protect user's data and personal information that may be collected in runtime. | High |
| S5 | All chatbot communications (Unity <-> Backend) must use secure protocols (HTTPS/TLS 1.2 or higher) to protect user input, logs, and retrieval-based data. | High |
| S6 | Chat transcripts, including short-term memory content, must be associated with anonymized session IDs or pseudonyms to comply with privacy regulations (e.g., GDPR). | High |


### **Scalability**
| ID   | Requirement Description | Priority |
|------|-------------------------|----------|
| Sc1 | The chatbot microservice must be scalable (e.g., containerization, load balancing) to handle sudden peaks in concurrent VR user sessions without significant increases in response times, in order to make it easier for further development down the road | Low |

### **Acessability**
| ID   | Requirement Description | Priority |
|------|-------------------------|----------|
| Ac1 | The chatbot must provide a simplified mode for users with cognitive challenges, offering shorter prompts and repeated instructions (per User Story 5.1). | Medium |
| Ac2 | The chatbot must work seamlessly with automatic teleportation or “auto-navigation” features for users with mobility challenges (e.g., offering a “Would you like me to teleport you there?” prompt). | Medium |


### **Safety**
| ID   | Requirement Description | Priority |
|------|-------------------------|----------|
| Sa1  | The system should not display any harmfull ideas or language.| Low |


## Business Requirements
*Business requirements represent the high-level needs and expectations of the business that the system must satisfy to achieve its purpose. They establish the strategic goals, objectives, and constraints that steer the system’s development and operation.*

### **Partnerships and Integrations**
| ID   | Requirement Description | Priority |
|------|-------------------------|----------|
| BR1 | Development should be in line with product owner (IMTEL NTNU), with continuous feedback and cooperation | High | 


### **End User Satisfaction**
| ID   | Requirement Description | Priority |
|------|-------------------------|----------|
| BR2.1 | End users should be satisfied with the chatbot | High |
| BR2.2 | User testing should be done do map the user's needs and satisfaction | High |

### **Compliance and Standards**
| ID   | Requirement Description | Priority |
|------|-------------------------|----------|
| BR3.1| The system must comply with GDPR and other relevant data protection regulations. | Medium |
| BR3.2| The system must comply with copyright laws | High | 

### **Cost Management**
| ID   | Requirement Description | Priority |
|------|-------------------------|----------|
| BR4 | The development of the IMTEL Chatbot must be carried out using low-cost or open-source solutions wherever possible to minimize expenditure and ensure the project remains within the allocated budget. This includes seeking free-tier or low-cost hosting options, avoiding expensive proprietary software licenses, and optimizing usage of any paid APIs or LLM services. | High| 





