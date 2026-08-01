# RAG Implementation and Cross-Encoder Implementation and Comparision

- Learnings :
  - Random embedding like TfidfVectorizer doesn't preserves the context. That's why we need Transformers for contextual embedding.
  - To retrive the related doucuments and indexing we used FIASS.
  - After retriving the related documents we might not get the related document at top so we need another tool to rank them.
  - Here comes Cross-Encoder that scores the documents retrived with the searching vector and score them.
  - Then it increases the chances to get the closest docuemnt to the searching vector, which helps to construct better prompt for classification of other purpose as requried.
  - Zero Shot Classification use and MAP@3 ranking.
