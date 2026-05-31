"""
Simulador de PLC OPC UA para pruebas sin hardware.

Expone un servidor OPC UA con nodos de clasificación que la app puede escribir.

Uso:
    python scripts/simular_plc.py
"""

import time
import sys

try:
    from opcua import Server, ua
except ImportError:
    print("Error: librería 'opcua' no instalada.")
    print("Ejecute: uv sync")
    sys.exit(1)


def main():
    server = Server()
    server.set_endpoint("opc.tcp://0.0.0.0:4840/pucp/")
    server.set_server_name("Simulador PLC PUCP")

    ns = server.register_namespace("PUCP")
    obj = server.nodes.objects.add_object(ns, "Clasificador")

    clasificacion = obj.add_variable(
        ns, "Clasificacion", ua.Variant(0, ua.VariantType.Int16),
    )
    confianza = obj.add_variable(
        ns, "Confianza", ua.Variant(0.0, ua.VariantType.Float),
    )

    clasificacion.set_writable()
    confianza.set_writable()

    server.start()

    print("=" * 55)
    print("  SIMULADOR PLC — OPC UA Server")
    print("=" * 55)
    print(f"  Endpoint: opc.tcp://localhost:4840/pucp/")
    print(f"  Namespace: ns={ns}")
    print(f"  Nodos:")
    print(f"    ns={ns};s=Clasificacion  (Int16)")
    print(f"    ns={ns};s=Confianza      (Float)")
    print()
    print("  Esperando datos... (Ctrl+C para salir)")
    print("=" * 55)
    print()

    prev_val = 0
    try:
        while True:
            val = clasificacion.get_value()
            if val != prev_val:
                conf = confianza.get_value()
                ts = time.strftime("%H:%M:%S")
                if val == 0:
                    print(f"  {ts}  ⚠ incierto (conf={conf:.1%})")
                else:
                    print(f"  {ts}  ✓ clase={val} (conf={conf:.1%})")
                prev_val = val
            time.sleep(0.3)
    except KeyboardInterrupt:
        print("\nDeteniendo servidor...")
    finally:
        server.stop()
        print("Servidor detenido.")


if __name__ == "__main__":
    main()
