#!/usr/bin/env python3
"""Create local TLS files and configure the Windows Gateway for a private LAN IP.

Requires the optional cryptography package. Generated keys and config are ignored.
Install the public lan-ca.crt on client devices to trust HTTPS for microphone access.
"""
import argparse
from datetime import datetime, timedelta, timezone
import ipaddress
import json
from pathlib import Path
import socket

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ip', required=True, type=ipaddress.IPv4Address)
    args = parser.parse_args()
    if not (args.ip.is_private and not args.ip.is_loopback and not args.ip.is_link_local
            and not args.ip.is_unspecified):
        parser.error('Use the computer\'s private LAN IPv4 address')
    config_path = ROOT / 'runtime/config.json'
    if not config_path.is_file():
        parser.error('Run scripts/configure.py --profile nvidia-rtx3060 first')
    config = json.loads(config_path.read_text(encoding='utf-8'))
    try:
        from cryptography import x509
        from cryptography.hazmat.primitives import hashes, serialization
        from cryptography.hazmat.primitives.asymmetric import rsa
        from cryptography.x509.oid import ExtendedKeyUsageOID, NameOID
    except ImportError:
        parser.error('Install the optional TLS dependency: python -m pip install cryptography')

    directory = ROOT / 'runtime/certs'
    directory.mkdir(exist_ok=True)
    now = datetime.now(timezone.utc)
    ca_path, ca_key_path = directory / 'lan-ca.crt', directory / 'lan-ca-key.pem'
    if ca_path.exists() != ca_key_path.exists():
        parser.error('Incomplete LAN CA files; restore the matching certificate/key pair')
    if ca_path.exists():
        ca = x509.load_pem_x509_certificate(ca_path.read_bytes())
        ca_key = serialization.load_pem_private_key(ca_key_path.read_bytes(), password=None)
        if ca.public_key().public_numbers() != ca_key.public_key().public_numbers():
            parser.error('LAN CA certificate and private key do not match')
        if ca.not_valid_after_utc <= now + timedelta(days=365):
            parser.error('LAN CA expires within a year; renew it and update client trust')
    else:
        ca_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        ca_name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, 'duplex.cpp local LAN CA')])
        ca = (x509.CertificateBuilder().subject_name(ca_name).issuer_name(ca_name)
              .public_key(ca_key.public_key()).serial_number(x509.random_serial_number())
              .not_valid_before(now - timedelta(minutes=5)).not_valid_after(now + timedelta(days=3650))
              .add_extension(x509.BasicConstraints(ca=True, path_length=0), critical=True)
              .add_extension(x509.KeyUsage(False, False, False, False, False, True, True, None, None), critical=True)
              .add_extension(x509.SubjectKeyIdentifier.from_public_key(ca_key.public_key()), critical=False)
              .sign(ca_key, hashes.SHA256()))
        ca_key_path.write_bytes(ca_key.private_bytes(serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8, serialization.NoEncryption()))
        ca_path.write_bytes(ca.public_bytes(serialization.Encoding.PEM))

    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, str(args.ip))])
    certificate = (x509.CertificateBuilder().subject_name(name).issuer_name(ca.subject)
        .public_key(key.public_key()).serial_number(x509.random_serial_number())
        .not_valid_before(now - timedelta(minutes=5)).not_valid_after(now + timedelta(days=365))
        .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
        .add_extension(x509.SubjectAlternativeName([x509.IPAddress(args.ip),
            x509.DNSName('localhost'), x509.DNSName(socket.gethostname())]), critical=False)
        .add_extension(x509.ExtendedKeyUsage([ExtendedKeyUsageOID.SERVER_AUTH]), critical=False)
        .add_extension(x509.KeyUsage(True, False, True, False, False, False, False, None, None), critical=True)
        .add_extension(x509.AuthorityKeyIdentifier.from_issuer_public_key(ca_key.public_key()), critical=False)
        .sign(ca_key, hashes.SHA256()))
    (directory / 'lan-key.pem').write_bytes(key.private_bytes(serialization.Encoding.PEM,
        serialization.PrivateFormat.PKCS8, serialization.NoEncryption()))
    (directory / 'lan-cert.pem').write_bytes(certificate.public_bytes(serialization.Encoding.PEM)
        + ca.public_bytes(serialization.Encoding.PEM))
    # Publish only the public CA. Private keys remain outside the static directory.
    (ROOT / 'runtime/static/lan-ca.crt').write_bytes(ca.public_bytes(serialization.Encoding.PEM))
    config['service'].update(gateway_host=str(args.ip), gateway_https=True,
        gateway_ssl_certfile='certs/lan-cert.pem', gateway_ssl_keyfile='certs/lan-key.pem')
    config_path.write_text(json.dumps(config, indent=2) + '\n', encoding='utf-8')
    port = config['service']['gateway_port']
    print(f'Configured: https://{args.ip}:{port}/omni')
    print(f'Client trust certificate: {ca_path}')
    print(f'CA SHA256: {ca.fingerprint(hashes.SHA256()).hex()}')
    print('Apply with scripts/windows/stop.ps1, then scripts/windows/start.ps1')


if __name__ == '__main__':
    main()
