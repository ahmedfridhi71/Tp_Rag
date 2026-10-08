import os
import re
import time
from typing import Optional

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from pypdf import PdfReader
from pydantic import BaseModel, Field


class InvoiceMetadata(BaseModel):
    invoice_code: Optional[str] = Field(default=None, description="The invoice number written after '#' (digits only, without '#'). Null if absent.")
    bill_to: Optional[str] = Field(default=None, description="The customer name found under 'Bill To'. Null if absent.")
    ship_to: Optional[str] = Field(default=None, description="The full shipping address found under 'Ship To' on one line. Null if absent.")
    total: Optional[float] = Field(default=None, description="The 'Total' amount of the invoice (same as Balance Due)")
    shipping: Optional[float] = Field(default=0.0, description="The 'Shipping' cost. 0 if there is no shipping line.")
    discount: Optional[float] = Field(default=0.0, description="The discount as a money amount (not the percentage). 0 if there is no discount line.")
    amount: Optional[float] = Field(default=None, description="The 'Subtotal': sum of the items before discount and shipping")


load_dotenv()

llm = ChatGroq(
    model="openai/gpt-oss-20b",
    groq_api_key=os.getenv("GROQ_API_KEY"),
    temperature=0,
)

structured_llm = llm.with_structured_output(InvoiceMetadata)

prompt = ChatPromptTemplate.from_template(
    """
    You are an invoice information extraction assistant.

    Extract: invoice_code, bill_to, ship_to, total, shipping, discount, amount.

    Layout of these invoices (very important):
    - The amounts appear first as a list of values, for example:
        $48.71 / $9.74 / $11.13 / $50.10
      and the labels appear AFTER, in the same order, for example:
        Subtotal / Discount (20%) / Shipping / Total
      Match the values to the labels by order.
    - If there is no "Discount" label, discount = 0.
    - If there is no "Shipping" label, shipping = 0.
    - amount = the Subtotal. total = the Total (equal to Balance Due).
    - discount must be the money amount, not the percentage.
    - Remove the "$" sign and thousands separators.
    - invoice_code is the number after "#" (without the "#").
    - Never return null for shipping or discount: use 0.
    - Do not invent information.

    Invoice text:
    {invoice_text}
    """
)

extractor_chain = prompt | structured_llm


def read_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    invoice_text = ""
    for page in reader.pages:
        text = page.extract_text()
        if text:
            invoice_text += text + "\n"
    return invoice_text


DATE_PATTERN = r"[A-Z][a-z]{2}\s+\d{1,2},?\s+\d{4}"


def clean(value):
    return " ".join(value.split())


def extract_metadata_regex(text):
    code = re.search(r"#\s*(\d+)", text)
    if not code:
        return InvoiceMetadata()

    bill = re.search(r"Bill To\s*:?\s*(.+?)\s*Ship To", text, re.S)
    ship = re.search(r"Ship To\s*:?\s*(.+?)\s*" + DATE_PATTERN, text, re.S)

    labels = [m.group(1) for m in re.finditer(r"^(Subtotal|Discount|Shipping|Total)\b", text, re.M)]
    first = re.search(r"^Subtotal\b", text, re.M)

    if not bill or not ship or not labels or not first:
        return None

    money = re.findall(r"\$([\d,]+\.\d{2})", text[:first.start()])
    if len(money) < len(labels):
        return None

    values = [float(x.replace(",", "")) for x in money[-len(labels):]]
    by_label = dict(zip(labels, values))

    return InvoiceMetadata(
        invoice_code=code.group(1),
        bill_to=clean(bill.group(1)),
        ship_to=clean(ship.group(1)),
        total=by_label.get("Total"),
        shipping=by_label.get("Shipping", 0.0),
        discount=by_label.get("Discount", 0.0),
        amount=by_label.get("Subtotal"),
    )


def is_consistent(meta):
    if meta.total is None or meta.amount is None:
        return False
    expected = round(meta.amount - (meta.discount or 0.0) + (meta.shipping or 0.0), 2)
    return abs(expected - meta.total) <= 0.05


def extract_metadata_llm(invoice_text, retries=3):
    last_error = None
    for attempt in range(retries):
        try:
            meta = extractor_chain.invoke({"invoice_text": invoice_text})
            meta.shipping = meta.shipping or 0.0
            meta.discount = meta.discount or 0.0
            return meta
        except Exception as e:
            last_error = e
            time.sleep(2 * (attempt + 1))
    raise last_error


def extract_metadata(invoice_text):
    meta = extract_metadata_regex(invoice_text)

    if meta is not None and meta.invoice_code is None:
        return meta

    if meta is not None and is_consistent(meta):
        return meta

    return extract_metadata_llm(invoice_text)


if __name__ == "__main__":
    pdf_path = os.path.join(
        "1000+ PDF_Invoice_Folder",
        "1000+ PDF_Invoice_Folder",
        "invoice_Aaron Bergman_36258.pdf",
    )

    result = extract_metadata_regex(read_pdf(pdf_path))

    print("Invoice code :", result.invoice_code)
    print("Bill to      :", result.bill_to)
    print("Ship to      :", result.ship_to)
    print("Total        :", result.total)
    print("Shipping     :", result.shipping)
    print("Discount     :", result.discount)
    print("Amount       :", result.amount)
    print("Coherent     :", is_consistent(result))

    print("\n=== ETAPE 4 OK ===")
