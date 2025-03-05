# Design Patterns Guide

This guide explains several common design patterns: **Factory**, **Abstract Factory**, **Facade**, **Data Access Object (DAO)**, **Data Transfer Object (DTO)**, and **Command**. For each pattern, you'll find an explanation of what it is, why it's used, and how to implement it, complete with code examples.

## Table of Contents
- [Factory Pattern](#factory-pattern)
- [Abstract Factory Pattern](#abstract-factory-pattern)
- [Facade Pattern](#facade-pattern)
- [Data Access Object (DAO) Pattern](#data-access-object-dao-pattern)
- [Data Transfer Object (DTO) Pattern](#data-transfer-object-dto-pattern)
- [Command Pattern](#command-pattern)

---

## Factory Pattern

### What It Is
The Factory pattern is a **creational design pattern** that provides an interface for creating objects, allowing subclasses to decide which class to instantiate. It centralizes object creation to avoid tightly coupling your code to specific classes.

### Why It's Used
- **Encapsulation of Object Creation:** Keeps the creation logic separate from the code that uses the objects.
- **Loose Coupling:** Reduces dependencies on concrete classes, making your code more flexible and easier to modify or extend.
- **Simplification:** Simplifies code by abstracting the instantiation process.

### How to Use It
1. **Define a common interface or abstract class** for the products.
2. **Create concrete classes** that implement the interface.
3. **Implement a factory class** with a method that returns an instance of the required concrete class based on input parameters.

#### Example
```python
class Product:
    def do_something(self):
        pass

class ConcreteProductA(Product):
    def do_something(self):
        return "Product A doing something!"

class ConcreteProductB(Product):
    def do_something(self):
        return "Product B doing something!"

class ProductFactory:
    @staticmethod
    def create_product(product_type: str) -> Product:
        if product_type == "A":
            return ConcreteProductA()
        elif product_type == "B":
            return ConcreteProductB()
        else:
            raise ValueError("Unknown product type")

# Usage
product = ProductFactory.create_product("A")
print(product.do_something())
```

## Abstract Factory Pattern
### What It Is

The Abstract Factory pattern is another creational design pattern that provides an interface for creating families of related or dependent objects without specifying their concrete classes.
Why It's Used

- Interchangeable Families: Enables the creation of a set of related objects that are designed to work together.
- Decoupling: Separates the code that uses the objects from the code that creates them.
- Consistency: Ensures that compatible products are used together.

### How to Use It

- Define abstract product interfaces for each type of product in the family.
- Define an abstract factory interface with methods to create each product.
- Implement concrete factories that produce a set of related products.
- Use the abstract factory in client code to create products without knowing their concrete classes.

### Example
```python
# Product interfaces
class Button:
    def render(self):
        pass

class Checkbox:
    def render(self):
        pass

# Abstract Factory
class GUIFactory:
    def create_button(self) -> Button:
        pass

    def create_checkbox(self) -> Checkbox:
        pass

# Concrete Products for Windows
class WindowsButton(Button):
    def render(self):
        return "Windows Button rendered"

class WindowsCheckbox(Checkbox):
    def render(self):
        return "Windows Checkbox rendered"

# Concrete Factory for Windows
class WindowsFactory(GUIFactory):
    def create_button(self) -> Button:
        return WindowsButton()

    def create_checkbox(self) -> Checkbox:
        return WindowsCheckbox()

# Usage
factory = WindowsFactory()
button = factory.create_button()
checkbox = factory.create_checkbox()
print(button.render(), checkbox.render())
```

## Facade Pattern
### What It Is

The Facade pattern is a structural design pattern that provides a simplified interface to a complex subsystem of classes, libraries, or frameworks.
Why It's Used

- Simplification: Reduces complexity by hiding the internal workings of a subsystem.
- Decoupling: Minimizes dependencies between the client and the subsystem, making the code easier to use and maintain.
- Ease of Use: Provides a single entry point for various functionalities.

How to Use It

- Create a facade class that wraps the complex subsystem.
- Expose simple methods in the facade that delegate tasks to the appropriate classes in the subsystem.
- Client code interacts with the facade instead of the complex subsystem directly.


### Example 
```python
class SubsystemA:
    def operation_a(self):
        return "SubsystemA: Operation A"

class SubsystemB:
    def operation_b(self):
        return "SubsystemB: Operation B"

class Facade:
    def __init__(self):
        self.subsystem_a = SubsystemA()
        self.subsystem_b = SubsystemB()

    def perform_operations(self):
        result_a = self.subsystem_a.operation_a()
        result_b = self.subsystem_b.operation_b()
        return f"{result_a} | {result_b}"

# Usage
facade = Facade()
print(facade.perform_operations())

```


## Data Access Object (DAO) Pattern
### What It Is

The Data Access Object (DAO) pattern is used to abstract and encapsulate all access to a data source. It separates the persistence layer from the business logic, providing a simple API for data operations.
Why It's Used

- Separation of Concerns: Keeps the business logic independent from data access code.
- Maintainability: Makes it easier to change the underlying data source without impacting the rest of the application.
- Testability: Facilitates unit testing by isolating database interactions.

### How to Use It

- Define a DAO interface that declares methods for data operations (e.g., CRUD operations).
- Implement the DAO interface in concrete classes that handle the specifics of the data source.
- Use the DAO in your business logic to perform data operations without coupling to the database code.

### Example
```python
class User:
    def __init__(self, id: int, name: str):
        self.id = id
        self.name = name

class UserDAO:
    def get_user(self, user_id: int) -> User:
        pass

    def add_user(self, user: User):
        pass

class UserDAOImpl(UserDAO):
    def __init__(self):
        self.users = {}

    def get_user(self, user_id: int) -> User:
        return self.users.get(user_id)

    def add_user(self, user: User):
        self.users[user.id] = user

# Usage
user_dao = UserDAOImpl()
user = User(1, "John Doe")
user_dao.add_user(user)
print(user_dao.get_user(1).name)

```

## Data Transfer Object (DTO) Pattern
### What It Is

The Data Transfer Object (DTO) pattern is used to transfer data between different layers or parts of a system. DTOs are simple objects that do not contain any business logic but only fields to hold data.
Why It's Used

- Efficiency: Reduces the number of remote calls by aggregating data.
- Clarity: Defines clear data contracts between layers or systems.
- Decoupling: Separates the data representation from business logic.

### How to Use It

- Create DTO classes that contain only the necessary data fields.
- Use DTOs to pass data between layers, such as from the data access layer to the presentation layer.
- Convert between domain models and DTOs as needed.

### Example
```python
class UserDTO:
    def __init__(self, id: int, name: str, email: str):
        self.id = id
        self.name = name
        self.email = email

# Usage
def get_user_dto() -> UserDTO:
    # In a real scenario, data would be fetched from a database or an API
    return UserDTO(1, "John Doe", "john.doe@example.com")

user_dto = get_user_dto()
print(user_dto.name, user_dto.email)

```

## Command Pattern
### What It Is

The Command pattern is a behavioral design pattern that encapsulates a request as an object, allowing for parameterization and queuing of requests. It separates the object that invokes the operation from the one that knows how to perform it.
Why It's Used

- Decoupling: Separates the request initiation from the execution.
- Flexibility: Supports operations like undo/redo, queuing, and logging.
- Reusability: Encapsulates operations as objects, making them easy to pass around and reuse.

### How to Use It

- Define a command interface with an execute method.
- Implement concrete command classes that encapsulate specific actions.
- Create an invoker class that holds and triggers the command.
- Use the command objects in client code to perform operations.

### Example
```python
class Command:
    def execute(self):
        pass

class Light:
    def on(self):
        return "Light is on"
    
    def off(self):
        return "Light is off"

class LightOnCommand(Command):
    def __init__(self, light: Light):
        self.light = light

    def execute(self):
        return self.light.on()

class LightOffCommand(Command):
    def __init__(self, light: Light):
        self.light = light

    def execute(self):
        return self.light.off()

class RemoteControl:
    def __init__(self):
        self.command = None

    def set_command(self, command: Command):
        self.command = command

    def press_button(self):
        if self.command:
            return self.command.execute()
        return "No command set"

# Usage
light = Light()
light_on_command = LightOnCommand(light)
remote = RemoteControl()
remote.set_command(light_on_command)
print(remote.press_button())

```

