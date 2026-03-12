# OpLib Lowering Specification

## ADDED Requirements

### Requirement: Interface-based OpLib lowering
The OpLib lowering mechanism SHALL be refactored to use an interface-driven approach to improve extensibility and maintainability.

#### Scenario: Replace hardcoded matching with interfaces
- A new `OpLibOpInterface` SHALL be defined to provide OpLib template information.
- The `PTOLowerToOpLibCalls` pass SHALL be refactored to use `OpLibOpInterface` instead of hardcoded operator matching.
- Existing PTO operators SHALL implement `OpLibOpInterface` to maintain functionality.
