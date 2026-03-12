# 🗄️ PostgreSQL Setup Guide

Application ippo fully **PostgreSQL** support pannum. Neenga commit pannuna apram matravanga (others) use panna intha steps-ah follow pannanum.

## 1. Prerequisites
System-la **PostgreSQL** installed-ah irukkanum and **psycopg2** library venum.
```bash
pip install psycopg2-binary
```

## 2. Environment Configuration (`.env`)
Unga `.env` file-la intha line-ah update pannunga. PostgreSQL password-ah correct-ah kudukkavum.
```env
DATABASE_URL=postgres://postgres:YOUR_PASSWORD@localhost:5432/YOUR_DB_NAME
```

## 3. Database Initialization
Pudhu database-la tables create panna intha command-ah oru thadava run pannanum:
```bash
.\venv\Scripts\python.exe -c "from app.db.session import init_db; init_db()"
```

## 4. Run Application
Inimel application-ah intha command use panni run pannunga:
```bash
.\venv\Scripts\python.exe run.py
```

## 💡 Important Notes:
- **Automatic Migration**: `app/db/session.py` file-la `.env` path search pannum logic ippo robust-ah irukku.
- **SQL Compatibility**: SQLite query-ah automatic-ah PostgreSQL query-ah convert panna connection wrapper add panni irukkom.
- **Terminal Log**: App start aagumbothu `>>> APPLICATION CONNECTED TO POSTGRESQL` message terminal-la varutha-nu verify pannikkonga.

---
*Created by Antigravity*
