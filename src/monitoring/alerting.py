"""Alerting module for sending notifications."""

import smtplib
from dataclasses import dataclass
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import List


@dataclass
class AlertConfig:
    """Configuration for alerts."""
    
    smtp_server: str
    smtp_port: int
    smtp_user: str
    smtp_password: str
    email_recipients: List[str]


class AlertManager:
    """Manage alerts and notifications."""
    
    def __init__(self, config: AlertConfig):
        self.config = config
    
    def send_email_alert(self, subject: str, message: str) -> bool:
        """Send an email alert."""
        try:
            msg = MIMEMultipart()
            msg["From"] = self.config.smtp_user
            msg["To"] = ", ".join(self.config.email_recipients)
            msg["Subject"] = subject
            
            msg.attach(MIMEText(message, "plain"))
            
            with smtplib.SMTP(self.config.smtp_server, self.config.smtp_port) as server:
                server.starttls()
                server.login(self.config.smtp_user, self.config.smtp_password)
                server.send_message(msg)
            
            return True
        except Exception as e:
            print(f"Failed to send alert: {e}")
            return False


if __name__ == "__main__":
    print("Alerting module loaded successfully")

