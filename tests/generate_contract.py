"""
Script to generate tests/sample_contracts/fake_contract.pdf
Run once with: python tests/generate_contract.py
"""
import os
import sys

import fitz  # PyMuPDF

CONTRACT_TEXT = """OFFER LETTER AND EMPLOYMENT AGREEMENT

TechCorp Solutions Pvt. Ltd.
42 Innovation Park, Bengaluru - 560001

Date: June 28, 2026

Dear Candidate,

We are pleased to offer you the position of Software Engineer at TechCorp Solutions Pvt. Ltd.

1. POSITION AND START DATE
Your position will be Software Engineer, reporting to the Engineering Manager.
Your anticipated start date is July 15, 2026.

2. COMPENSATION
You will receive an annual base salary of Rs. 12,00,000 (Rupees Twelve Lakhs), paid monthly.
Performance bonuses are at the sole discretion of the Company and are not guaranteed.

3. PROBATION PERIOD
You will be on probation for a period of six (6) months from your date of joining.
During this period, either party may terminate employment with 7 days written notice.
The Company reserves the right to extend the probation period at its sole discretion.

4. INTELLECTUAL PROPERTY ASSIGNMENT
You agree that all inventions, software, code, ideas, designs, processes, and works
of authorship that you create, conceive, or develop during your employment — whether
during or outside of working hours, and whether or not using Company resources —
shall be the sole and exclusive property of TechCorp Solutions Pvt. Ltd.
You irrevocably assign all such intellectual property rights to the Company.
This clause survives the termination of this agreement.

5. NON-COMPETE CLAUSE
For a period of two (2) years following the termination of your employment, for any
reason whatsoever, you agree not to directly or indirectly engage in, work for,
advise, or have any financial interest in any business that competes with the Company
anywhere in India or globally. Violation of this clause entitles the Company to seek
injunctive relief and damages of not less than Rs. 50,00,000 (Rupees Fifty Lakhs).

6. NON-SOLICITATION
For a period of two (2) years after your employment ends, you shall not solicit,
hire, or attempt to hire any employee or contractor of TechCorp Solutions, nor
shall you solicit any client, customer, or vendor of the Company.

7. CONFIDENTIALITY
You agree to maintain strict confidentiality of all Company information, trade secrets,
business strategies, and client data, both during and after employment, for an
unlimited period of time. Breach of this clause shall result in liquidated damages
of Rs. 25,00,000 (Rupees Twenty-Five Lakhs) per incident.

8. INDEMNIFICATION
You agree to indemnify and hold harmless TechCorp Solutions, its officers, directors,
and employees from any and all claims, damages, liabilities, costs, and expenses
arising out of or related to your actions during employment, without any limitation
on the amount of indemnification.

9. MANDATORY ARBITRATION
Any dispute arising under this agreement shall be resolved exclusively through
binding private arbitration in Bengaluru, under the rules of the Indian Arbitration
and Conciliation Act. You waive your right to a jury trial or class action.
The arbitrator's decision shall be final and not subject to appeal.

10. UNILATERAL AMENDMENT
The Company reserves the right to modify the terms and conditions of your employment,
including compensation, benefits, role, and responsibilities, at any time and at its
sole discretion, with 30 days written notice to you.

11. GOVERNING LAW AND JURISDICTION
This agreement is governed by the laws of the State of Karnataka, India.
All disputes shall be subject to the exclusive jurisdiction of courts in Bengaluru.

12. AT-WILL EMPLOYMENT AND TERMINATION
Notwithstanding the probation terms, the Company may terminate your employment at
any time, with or without cause, and with or without notice, at its sole discretion.
Upon termination, you shall return all Company property within 24 hours.

Please sign and return this offer letter within 3 days of receipt.

Accepted and agreed:

____________________             ____________________
Candidate Signature              Authorised Signatory
Date: _______________            TechCorp Solutions Pvt. Ltd.
"""

OUTPUT = os.path.join(
    os.path.dirname(__file__),
    "sample_contracts",
    "fake_contract.pdf",
)

os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)

doc = fitz.open()
page = doc.new_page(width=595, height=842)  # A4

y = 60
for line in CONTRACT_TEXT.strip().splitlines():
    # Bold style for numbered headings
    fontsize = 9
    if line and line[0].isdigit() and ". " in line[:5]:
        fontsize = 10
    elif line.isupper() and len(line) > 3:
        fontsize = 11

    page.insert_text((50, y), line, fontsize=fontsize, color=(0, 0, 0))
    y += fontsize + 4

    if y > 800:
        page = doc.new_page(width=595, height=842)
        y = 60

doc.save(OUTPUT)
doc.close()

print(f"Created: {OUTPUT}")
print(f"Size:    {os.path.getsize(OUTPUT):,} bytes")

# Verify with our parser
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from mcp_server.tools.parse_document import parse_document
from mcp_server.tools.extract_clauses import extract_clauses

result = parse_document(OUTPUT)
clauses = extract_clauses(result["text"])

print(f"Pages:   {result['page_count']}")
print(f"Chars:   {len(result['text']):,}")
print(f"Clauses: {len(clauses)}")
print()
print("=== Clause headings ===")
for c in clauses:
    print(f"  [{c['position']}] {c['heading'][:80]}")
