---
name: Feature request
description: Propose a focused improvement to Model Lab
labels: [enhancement]
body:
  - type: textarea
    id: problem
    attributes:
      label: Problem or opportunity
      description: What user or engineering problem does this solve?
    validations:
      required: true
  - type: textarea
    id: proposal
    attributes:
      label: Proposed change
      description: Describe the smallest coherent implementation that addresses the problem.
    validations:
      required: true
  - type: textarea
    id: validation
    attributes:
      label: Validation
      description: What tests, contracts, or runtime evidence should prove this change works?
    validations:
      required: true
