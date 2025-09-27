# TODO: Implementing Health Report Dashboard and Diet/Lifestyle Suggestions

## Overview
Enhance the prediction result section with a visual dashboard using Chart.js (pie chart for risk probability, bar chart for key health metrics). Add rule-based diet and lifestyle suggestions for high-risk cases (probability > 50%). All changes in `backend/frontend/index.html`. Maintain existing design aesthetics (gradients, animations, responsiveness).

## Steps to Complete

### 1. Add Chart.js CDN and New Elements to HTML Structure
- Include Chart.js script tag in `<head>`.
- In `#result` div: 
  - Wrap existing text in a header section.
  - Add `<canvas id="risk-pie-chart"></canvas>` for pie chart (risk vs. no-risk %).
  - Add `<canvas id="metrics-bar-chart"></canvas>` for bar chart (BP, Cholesterol, BMI values with thresholds for color-coding: green < normal, red > high).
  - Add `<div id="suggestions" class="hidden card">` with header "Personalized Health Recommendations" and list of tips (e.g., ul with li for diet/exercise based on inputs like high BMI, smoking).
- Use Font Awesome icons (e.g., fa-apple-alt for diet, fa-running for exercise).

### 2. Add CSS for New Dashboard Elements
- Style charts: Container divs with flex/grid layout (pie left, bar right on desktop; stacked on mobile).
- Suggestions card: Match glassmorphism (rgba background, blur), collapsible with toggle button if needed, green accent for positive advice.
- Responsive: Charts scale with `max-width: 100%; height: auto;`.
- Colors: Pie - red for risk, green for safe; Bar - dynamic (e.g., BP >140 red, else green).

### 3. Update JavaScript for Rendering
- In prediction handler (after fetch `/predict`):
  - Store `data.prediction`, `data.probability`, and `features` (from formData).
  - Update text as before.
  - Render pie chart: Use Chart.js to create pie with labels ["Risk", "No Risk"], data [probability*100, (1-probability)*100], colors ['#f44336', '#4caf50'].
  - Render bar chart: Data for labels ["Blood Pressure", "Cholesterol", "BMI"], values from features, background colors based on thresholds (e.g., BP>140 red, Cholesterol>200 red, BMI>25 red).
  - For line chart (BP/sugar over time): Since no history, mock simple line with 3 points (e.g., average low, medium, current BP) or integrate as second chart if space; simplify to bar for now.
  - If probability > 0.5 (high risk): Show `#suggestions` with dynamic rules:
    - If Smoking="Yes": "Quit smoking immediately to reduce risk by 30%."
    - If BMI>25: "Adopt a low-carb diet and exercise 30min/day."
    - If BP>140: "Follow DASH diet (low sodium), monitor daily."
    - If Cholesterol>200: "Incorporate omega-3 rich foods like fish/nuts."
    - General: "Consult a doctor; regular check-ups recommended."
  - Add error handling for chart rendering.

### 4. Testing and Verification
- Run `python backend/app.py`.
- Test low-risk inputs (e.g., Age 30, Female, BP 120, Cholesterol 150, BMI 22, No): Verify charts show low risk pie, green bars, no suggestions.
- Test high-risk (Age 70, Male, BP 180, Cholesterol 300, BMI 35, Yes): Verify red pie slice, red bars, suggestions appear with relevant tips.
- Check mobile: Charts responsive, suggestions readable.
- Edge cases: Invalid inputs (handled by form), probability=0.5 (borderline, show suggestions?).
- Visual: Ensure matches existing design (no breakage in animations, gradients).

### 5. Final Polish
- Add tooltips to charts (Chart.js default).
- Ensure accessibility: ARIA labels for canvases, alt text equivalents.
- Update result background dynamically: Green for low, red/orange for high.

Progress: [ ] Step 1 [ ] Step 2 [ ] Step 3 [ ] Step 4 [ ] Step 5
