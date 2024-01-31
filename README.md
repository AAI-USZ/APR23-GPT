# APR23-GPT

candidate_files directory: contains the closest candidate by edit distance for each model and each bug.
For each model, there is a file containing the candidates that were generated with each model.
  - The number is an id
  - The next line indicates which project and which file contains the bug we are trying to fix
  - The 3rd line is the developer patch we are trying to generate
  - The 4th line is the generated patch that is closest to the developer patch by edit distance

models directory: contains the source code for each model
