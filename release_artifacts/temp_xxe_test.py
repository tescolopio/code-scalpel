# XXE Vulnerability Trap File for v1.4.0 Testing
# CWE-611: Improper Restriction of XML External Entity Reference

import xml.etree.ElementTree as ET
from flask import Flask, request

app = Flask(__name__)


@app.route("/parse_xml")
def parse_xml():
    """Vulnerable: Parses user-supplied XML with dangerous parser."""
    user_xml = request.args.get("xml")
    
    # VULNERABLE: Direct XML parsing of user input
    tree = ET.parse(user_xml)  # XXE SINK - should be detected
    root = tree.getroot()
    
    return f"Parsed: {root.tag}"


@app.route("/parse_string")
def parse_string():
    """Vulnerable: Parses user-supplied XML string."""
    xml_data = request.data
    
    # VULNERABLE: fromstring with user input
    root = ET.fromstring(xml_data)  # XXE SINK - should be detected
    
    return f"Root tag: {root.tag}"
