## Pipelines

This document contains notes made during development and ideas on how pipelines are going to work 

### Main Idea

We want to explore the different compilation pipeline steps to see which will produce the best final circuit.
In order to do this, we need a way to represent a composition of steps in some process.

We will have a "Raw" synthesis method (XAG, Truth Table, and Classiq) which will produce some quantum circuit.
We then define a simple PipelineStep interface which performs some operation on the quantum circuit.
An implementor of the PipelineStep interface has a transform() function which takes a quantum circuit, kwargs, and outputs a quantum circuit.
This is supposed to make each step composable and swappable in the order, as long as the interface is followed.

We define a Compiler as a Synthesizer and a list of steps.
A Synthesizer should accept a ProblemInstance and synthesize accordingly (already handled for Clique ProblemInstances).
After synthesis, each step is performed in the order they appear when the compiler was defined. 
For a basic naming convention, Compilers should be named "Synthesizer+Step+Step+...+Step" to be filled in by the names of each class. 

### Refactoring the rest of the library 

#### Data separation
Currently, problem instances are stored together with trials. 
This is not a good solution, and it makes it hard to use the same problems, because I have to copy the whole database and delete the trials table. 
Instead, I think we should just store a database of problem instances in the library under `shared/problems/clique_problems.db`. 
The library can define a relative path so it is always loaded when the library is loaded, and accessing the problems is very easy.
We can use foriegn keys to keep track of which problem we referenced in our separate trials database.
The trails database can be stored wherever, and that way experiments can share the same problems without having to duplicate them.

#### New Compiler benchmarking, storage, and approach

After the problems have been decoupled, we need a more robust way to store the results of our compiler benchmarking.
The new compiler interface opens many possibilities for various methods of compilation, and we need a strategy that supports that.
I want to track high and low level statistics.
High level statisics are gathered from the circuit produced DIRECTLY BEFORE transpilation.
This indicates a logical structure, but does not inform us much on how it is implemented in hardware, which is critical for determining performance. 
Low level statistics are gathered from the circuit FOLLOWING transpilation.
We gather the same information, but on the transpiled circuit.
The key metrics to track for all of them:
  - end to end compilation time
  - number of qubits 
  - depth (high and low)
  - entangling gate count (this is tricky) (high and low)
  - possibly more in the future

Approach: I want to use large parallelism to run these large scripts in parallel.
I want to take a job from the command line, turn it in to a list of smaller jobs, and then use some kind of worker/thread/task pool to dispatch jobs to workers.
This minimizes the number of cores that aren't working.
I want to set memory limits per worker with the default being free memory - 4GB / number of workers.
The benchmark should be a python script.

Tasks: In Order of Priority

- Implement the compilation interfaces (Compilers and Steps) as well as an easy way to serialize them for storage in a database. 
  - Serialization/Deserialization for easy reproducability
  - Simple interface (Quantum Circuit -> transform() -> Quantum Circuit)
- New storage interface for Compiler 
- New benchmark script
- Split up the problem storage from Trial storage, making problems easily accessible and keeping the same patterns for searching/filtering. 
- Create Special Trials database for storing the results of hardware experiments
