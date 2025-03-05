
# **Architectural Design**

At one point we were discussing using a multi-tier architecture, specifically an three-tier model, as we could divide the chatbot system into three tiers; presentation tier, application-tier and data access tier. However, since we are working on a large code base where there are no explicit separation between the business/application layer and the presentation tier, this proved to be a poor choice when integrating our solution as we implement the VR chatbot system. Therefore we opted for a microservice architecture, with approval from the product owner. 

## **MicroService Architecture**

Microservice architecture is a software design approach where a larger application is built as a collection of small, independently deployable services. These are referred to as microservices. Each microservice runs in its own process, communicates over HTTPS (in our case, but often uses a more lightweight protocol like HTTP), and focuses on one specific capability or business function. In addition to various benefits of this architecture, some of the members of this group already have experience with this architectural approach.

These are the core concepts of this architecture:
### **Single responsibility** 
Each microservice is designed to perform one main function, keeping the codebase for each service small and easier to maintain. As for development, with microservices, teams can work in parallel on different services.

### **Independent Deployment**
Services can be refactored, redeployed, or scaled without impacting other parts of the system. For instance, if you update the chatbot’s language model, you don’t have to redeploy your other components. This also makes the codebase more manageable for other developers after we finish this project. Another benefit with independent deployment is that failure in one microservice doesn’t necessarily bring down the entire system.

### **Loose Coupling and Autonomy**
Microservices communicate through well-defined APIs, and do not depend on each other’s internal implementations. This separation means you can mix technologies. The current codebase is mostly written in C#, but due to python's nature of fast development, as well as it's well documented stance in artificial intelligence, we want to write separated services in python for the chatbot application. This architecture enable us to do interchange languages seamlessly. 








