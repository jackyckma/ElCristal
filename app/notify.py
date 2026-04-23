from __future__ import annotations

import logging
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

from app import config

logger = logging.getLogger(__name__)


def send_completion_email(
    recipient: str,
    job_id: str,
    file_names: list[str],
    download_urls: list[str],
) -> None:
    """Send job-completion email with download links to the user."""
    if not config.SMTP_HOST or not config.SMTP_USER:
        logger.warning("SMTP not configured — skipping notification for job %s", job_id)
        return

    subject = f"ElCristal — Your restored tango tracks are ready"

    file_list_html = "\n".join(
        f'<li><a href="{url}">{name}</a></li>'
        for name, url in zip(file_names, download_urls)
    )
    file_list_plain = "\n".join(
        f"  • {name}: {url}" for name, url in zip(file_names, download_urls)
    )

    html_body = f"""\
<html><body>
<p>Your ElCristal audio restoration job is complete.</p>
<p>Download your restored tracks (links expire in {config.OUTPUT_TTL_HOURS} hours):</p>
<ul>{file_list_html}</ul>
<p><em>ElCristal — Tango Audio Restoration</em><br>
<a href="https://github.com/jackyckma/elcristal">github.com/jackyckma/elcristal</a></p>
</body></html>"""

    plain_body = (
        f"Your ElCristal audio restoration job is complete.\n\n"
        f"Download your restored tracks (links expire in {config.OUTPUT_TTL_HOURS} hours):\n"
        f"{file_list_plain}\n\n"
        f"ElCristal — Tango Audio Restoration\n"
        f"https://github.com/jackyckma/elcristal"
    )

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = config.SMTP_USER
    msg["To"] = recipient
    msg.attach(MIMEText(plain_body, "plain"))
    msg.attach(MIMEText(html_body, "html"))

    try:
        with smtplib.SMTP(config.SMTP_HOST, config.SMTP_PORT) as server:
            server.ehlo()
            server.starttls()
            server.login(config.SMTP_USER, config.SMTP_PASS)
            server.sendmail(config.SMTP_USER, recipient, msg.as_string())
        logger.info("Completion email sent to %s for job %s", recipient, job_id)
    except Exception:
        logger.exception("Failed to send completion email for job %s", job_id)
