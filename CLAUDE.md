Review code base and README.md to get a good understanding of the current state
Let us now focus on Smart Vector Store Management, unless a document changes, there is no need to keep on embedding it each time we are running a querry
Please share your plan on how to do this with a focus on latency.  We also do not want the stores to become too large and so should be purging it after it becomes a certain size
From a future functionality POV - we should also design in such a way that we can compare documents
Please share your strategy, get feedback and then implement
Develop unit tests and run them
Complete regression testing after unit tests pass
Note that test_baseline.py takes greater than 5 minutes to run and so for regression testing purposes, just run one of the questions
Remove any temporary files created
Please update README.md to reflect all changes.  Please also review the project structure section to make sure it is up to date
Recommend your view on what I should do next to improve latency