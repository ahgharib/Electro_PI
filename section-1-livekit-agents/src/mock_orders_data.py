"""The mock order data. THIS is the file to edit to add/remove test orders.

Nothing in this file talks to the LLM, tools, or the LiveKit SDK -- it's
plain Python data, kept separate from persona.py on purpose so "where's
the data" has one obvious answer.

To add an order: add a line to ORDER_SEED below.
To remove an order: delete its line.
To change an order's status/ETA/cancellability: edit its line.

Each `SupportAgent` instance gets its own fresh copy of this dict (see
`seed_orders()` and `SupportAgent.__init__` in persona.py) -- editing
ORDER_SEED changes the starting data for every new conversation; it does
not affect orders already cancelled during a conversation that's already
running.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Order:
    order_id: str
    status: str
    eta_minutes: int | None
    cancellable: bool


# order_id -> Order. Add/remove/edit entries here.
ORDER_SEED: dict[str, Order] = {
    "A100": Order("A100", "preparing", eta_minutes=25, cancellable=True),
    "A101": Order("A101", "out_for_delivery", eta_minutes=8, cancellable=False),
    "A102": Order("A102", "delivered", eta_minutes=None, cancellable=False),
    "A103": Order("A103", "cancelled", eta_minutes=None, cancellable=False),
}


def seed_orders() -> dict[str, Order]:
    """A fresh copy of ORDER_SEED for one SupportAgent/conversation."""
    return {order_id: Order(**vars(order)) for order_id, order in ORDER_SEED.items()}
