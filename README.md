#  Email Spam Detection System – Frontend (Next.js + TailwindCSS)

This is the **frontend application** for an AI-powered email spam detection system. It analyzes the subject line of email content using a **BERT model** served through a **FastAPI backend**, and delivers real-time predictions with rich UI feedback.

---

##  Features

 Paste full email content; subject is auto-extracted
 Uses AI to detect spam/ham with confidence score
 Real-time response with average latency < 1s
 Modern animated UI using TailwindCSS, Lucide, and custom components
 Displays risk levels, confidence percentage, and reasons for classification

---

Tech Stack

| Layer         | Technology                  |
|---------------|-----------------------------|
| Frontend      | Next.js (React)             |
| UI Framework  | TailwindCSS                 |
| Icons         | [Lucide React](https://lucide.dev/icons) |
| Backend API   | FastAPI (Python)            |
| Model         | Fine-tuned BERT             |

---



##  Local Setup

### 1. Clone the repository:

```bash
git clone https://github.com/your-username/email-spam-frontend.git
cd email-spam-frontend
2. Install dependencies:
Using pnpm:


pnpm install
Or using npm:


npm install
3. Run development server:

pnpm dev
# or
npm run dev
Then open http://localhost:3000

 Backend API Connection
This frontend is connected to the backend at:


http://127.0.0.1:8000/predict
Make sure to run your FastAPI backend server before using this app.

Expected API Input:

json
Copy
Edit
{
  "subject": "Your subject line here"
}
Expected API Response:


{
  "subject": "Your subject",
  "prediction": "spam",
  "confidence": 0.93
}
