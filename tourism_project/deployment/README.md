---
title: Score a Lead
emoji: 🧳
colorFrom: green
colorTo: blue
sdk: docker
app_port: 8501
pinned: false
short_description: Which Wellness Tourism leads to contact first
---

# Score a Lead

Streamlit app for the marketing team of *Visit with Us*. A lead is a customer who has not been
contacted yet. Enter the profile of one lead and see whether that lead belongs to the group most
likely to buy the Wellness Tourism Package, so the sales team can decide whom to contact first.

- The prediction uses only information known **before** a customer is contacted.
- The model is a random forest selected and evaluated in the project's MLOps pipeline; it is loaded
  from a private Hugging Face model repository at a fixed version (`MODEL_REVISION`).
- The app shows a ranking score and the contact group the lead falls into, together with how
  well that group performed on held-out test customers. The score is not a probability.

The front matter above configures this Space: `sdk: docker` builds the included `Dockerfile`, and
`app_port: 8501` routes traffic to Streamlit's port.
