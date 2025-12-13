#!/usr/bin/env python3
"""
Code Scalpel Demo: Refactoring with Surgical Extraction
========================================================

This demo shows the dramatic difference between naive full-file context
and Code Scalpel's surgical extraction when preparing context for an LLM
refactoring task.

Scenario: Refactor the `calculate_tax` function to support multiple tax rates.
"""

import sys
from pathlib import Path

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.code_scalpel.code_analyzer import CodeAnalyzer
from src.code_scalpel.surgical_extractor import SurgicalExtractor


# =============================================================================
# Sample Codebase (simulating a real project)
# =============================================================================

ECOMMERCE_CODE = '''
"""E-commerce order processing module."""

from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional, Dict
from decimal import Decimal
import json
import logging

logger = logging.getLogger(__name__)


# =============================================================================
# Product Management
# =============================================================================

@dataclass
class Product:
    """Represents a product in the catalog."""
    id: str
    name: str
    price: Decimal
    category: str
    inventory_count: int
    description: str = ""
    is_active: bool = True
    created_at: datetime = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.utcnow()

    def is_in_stock(self) -> bool:
        """Check if product is available."""
        return self.inventory_count > 0 and self.is_active

    def reserve_inventory(self, quantity: int) -> bool:
        """Reserve inventory for an order."""
        if quantity <= self.inventory_count:
            self.inventory_count -= quantity
            return True
        return False

    def release_inventory(self, quantity: int):
        """Release reserved inventory."""
        self.inventory_count += quantity

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "name": self.name,
            "price": str(self.price),
            "category": self.category,
            "inventory_count": self.inventory_count,
            "is_active": self.is_active
        }


class ProductCatalog:
    """Manages the product catalog."""

    def __init__(self):
        self.products: Dict[str, Product] = {}

    def add_product(self, product: Product):
        """Add a product to the catalog."""
        self.products[product.id] = product
        logger.info(f"Added product: {product.name}")

    def get_product(self, product_id: str) -> Optional[Product]:
        """Get product by ID."""
        return self.products.get(product_id)

    def search_products(self, query: str) -> List[Product]:
        """Search products by name or description."""
        query = query.lower()
        return [p for p in self.products.values()
                if query in p.name.lower() or query in p.description.lower()]

    def get_by_category(self, category: str) -> List[Product]:
        """Get all products in a category."""
        return [p for p in self.products.values() if p.category == category]


# =============================================================================
# Shopping Cart
# =============================================================================

@dataclass
class CartItem:
    """An item in the shopping cart."""
    product: Product
    quantity: int

    @property
    def subtotal(self) -> Decimal:
        """Calculate item subtotal."""
        return self.product.price * self.quantity


class ShoppingCart:
    """Shopping cart for a customer."""

    def __init__(self, customer_id: str):
        self.customer_id = customer_id
        self.items: List[CartItem] = []
        self.created_at = datetime.utcnow()

    def add_item(self, product: Product, quantity: int = 1):
        """Add item to cart."""
        # Check if product already in cart
        for item in self.items:
            if item.product.id == product.id:
                item.quantity += quantity
                return

        self.items.append(CartItem(product=product, quantity=quantity))

    def remove_item(self, product_id: str):
        """Remove item from cart."""
        self.items = [i for i in self.items if i.product.id != product_id]

    def update_quantity(self, product_id: str, quantity: int):
        """Update item quantity."""
        for item in self.items:
            if item.product.id == product_id:
                if quantity <= 0:
                    self.remove_item(product_id)
                else:
                    item.quantity = quantity
                return

    def get_subtotal(self) -> Decimal:
        """Calculate cart subtotal before tax."""
        return sum(item.subtotal for item in self.items)

    def clear(self):
        """Clear all items from cart."""
        self.items = []


# =============================================================================
# Tax Calculation (TARGET FOR REFACTORING)
# =============================================================================

TAX_RATE = Decimal("0.08")  # 8% tax rate

def calculate_tax(subtotal: Decimal) -> Decimal:
    """
    Calculate tax on a subtotal.

    THIS IS THE FUNCTION WE WANT TO REFACTOR.
    We need to add support for multiple tax rates based on:
    - Customer location (state)
    - Product category (some categories tax-exempt)
    """
    return subtotal * TAX_RATE


def calculate_order_total(cart: ShoppingCart) -> Dict[str, Decimal]:
    """Calculate complete order total with tax."""
    subtotal = cart.get_subtotal()
    tax = calculate_tax(subtotal)
    total = subtotal + tax

    return {
        "subtotal": subtotal,
        "tax": tax,
        "total": total
    }


# =============================================================================
# Order Processing
# =============================================================================

@dataclass
class Address:
    """Customer address."""
    street: str
    city: str
    state: str
    zip_code: str
    country: str = "USA"

    def format(self) -> str:
        return f"{self.street}, {self.city}, {self.state} {self.zip_code}"


@dataclass
class Customer:
    """Customer information."""
    id: str
    email: str
    name: str
    shipping_address: Optional[Address] = None
    billing_address: Optional[Address] = None

    def has_complete_address(self) -> bool:
        return self.shipping_address is not None


class OrderStatus:
    PENDING = "pending"
    CONFIRMED = "confirmed"
    PROCESSING = "processing"
    SHIPPED = "shipped"
    DELIVERED = "delivered"
    CANCELLED = "cancelled"


@dataclass
class Order:
    """A customer order."""
    id: str
    customer: Customer
    items: List[CartItem]
    subtotal: Decimal
    tax: Decimal
    total: Decimal
    status: str = OrderStatus.PENDING
    created_at: datetime = None
    shipping_address: Address = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.utcnow()

    def confirm(self):
        """Confirm the order."""
        self.status = OrderStatus.CONFIRMED
        logger.info(f"Order {self.id} confirmed")

    def ship(self, tracking_number: str):
        """Mark order as shipped."""
        self.status = OrderStatus.SHIPPED
        self.tracking_number = tracking_number

    def cancel(self):
        """Cancel the order."""
        self.status = OrderStatus.CANCELLED
        # Release inventory
        for item in self.items:
            item.product.release_inventory(item.quantity)


class OrderService:
    """Service for creating and managing orders."""

    def __init__(self, catalog: ProductCatalog):
        self.catalog = catalog
        self.orders: Dict[str, Order] = {}

    def create_order(self, customer: Customer, cart: ShoppingCart) -> Optional[Order]:
        """Create an order from a shopping cart."""
        if not cart.items:
            logger.error("Cannot create order from empty cart")
            return None

        if not customer.has_complete_address():
            logger.error("Customer address incomplete")
            return None

        # Reserve inventory
        for item in cart.items:
            if not item.product.reserve_inventory(item.quantity):
                logger.error(f"Insufficient inventory for {item.product.name}")
                return None

        # Calculate totals using the tax calculation
        totals = calculate_order_total(cart)

        order = Order(
            id=self._generate_order_id(),
            customer=customer,
            items=cart.items.copy(),
            subtotal=totals["subtotal"],
            tax=totals["tax"],
            total=totals["total"],
            shipping_address=customer.shipping_address
        )

        self.orders[order.id] = order
        cart.clear()

        return order

    def get_order(self, order_id: str) -> Optional[Order]:
        """Get order by ID."""
        return self.orders.get(order_id)

    def _generate_order_id(self) -> str:
        """Generate unique order ID."""
        import uuid
        return f"ORD-{uuid.uuid4().hex[:8].upper()}"


# =============================================================================
# Discount System
# =============================================================================

@dataclass
class Discount:
    """A discount or coupon."""
    code: str
    percentage: Decimal
    min_purchase: Decimal = Decimal("0")
    max_uses: int = None
    current_uses: int = 0

    def is_valid(self, subtotal: Decimal) -> bool:
        """Check if discount can be applied."""
        if subtotal < self.min_purchase:
            return False
        if self.max_uses and self.current_uses >= self.max_uses:
            return False
        return True

    def calculate_discount(self, subtotal: Decimal) -> Decimal:
        """Calculate discount amount."""
        if not self.is_valid(subtotal):
            return Decimal("0")
        return subtotal * (self.percentage / 100)


class DiscountService:
    """Service for managing discounts."""

    def __init__(self):
        self.discounts: Dict[str, Discount] = {}

    def add_discount(self, discount: Discount):
        """Add a discount code."""
        self.discounts[discount.code] = discount

    def apply_discount(self, code: str, subtotal: Decimal) -> Decimal:
        """Apply a discount code and return the discount amount."""
        discount = self.discounts.get(code)
        if discount:
            return discount.calculate_discount(subtotal)
        return Decimal("0")


# =============================================================================
# Reporting
# =============================================================================

class SalesReport:
    """Generate sales reports."""

    def __init__(self, order_service: OrderService):
        self.order_service = order_service

    def total_sales(self) -> Decimal:
        """Calculate total sales."""
        return sum(o.total for o in self.order_service.orders.values())

    def total_tax_collected(self) -> Decimal:
        """Calculate total tax collected."""
        return sum(o.tax for o in self.order_service.orders.values())

    def orders_by_status(self) -> Dict[str, int]:
        """Count orders by status."""
        counts = {}
        for order in self.order_service.orders.values():
            counts[order.status] = counts.get(order.status, 0) + 1
        return counts
'''


def demo_without_code_scalpel():
    """Show what context an LLM would receive WITHOUT Code Scalpel."""
    print("=" * 70)
    print("SCENARIO: WITHOUT CODE SCALPEL")
    print("=" * 70)
    print()
    print("Task: Refactor calculate_tax() to support multiple tax rates")
    print()
    print("The LLM receives the ENTIRE FILE as context:")
    print("-" * 70)

    lines = ECOMMERCE_CODE.strip().split('\n')
    total_chars = len(ECOMMERCE_CODE)
    total_tokens = total_chars // 4  # Rough estimate

    print(f"Total lines: {len(lines)}")
    print(f"Total characters: {total_chars:,}")
    print(f"Estimated tokens: {total_tokens:,}")
    print()
    print("Problems with this approach:")
    print("1. Context window pollution - 90%+ of the code is irrelevant")
    print("2. LLM may get confused by unrelated classes/functions")
    print("3. Waste of tokens that could be used for reasoning")
    print("4. May accidentally modify unrelated code")
    print()

    return {
        "lines": len(lines),
        "chars": total_chars,
        "tokens": total_tokens
    }


def demo_with_code_scalpel():
    """Show what context an LLM would receive WITH Code Scalpel."""
    print("=" * 70)
    print("SCENARIO: WITH CODE SCALPEL")
    print("=" * 70)
    print()
    print("Task: Refactor calculate_tax() to support multiple tax rates")
    print()
    print("Code Scalpel extracts ONLY the relevant code:")
    print("-" * 70)

    # Use Code Scalpel to extract
    extractor = SurgicalExtractor(ECOMMERCE_CODE)

    # Extract the target function
    result = extractor.get_function("calculate_tax")

    # Also extract its direct caller for context
    caller_result = extractor.get_function("calculate_order_total")

    # Build the surgical context
    surgical_context = []

    # Add imports that would be needed
    surgical_context.append("# Required imports")
    surgical_context.append("from decimal import Decimal")
    surgical_context.append("")

    # Add the TAX_RATE constant (which calculate_tax uses)
    surgical_context.append("# Current tax configuration")
    surgical_context.append('TAX_RATE = Decimal("0.08")  # 8% tax rate')
    surgical_context.append("")

    # Add the target function
    surgical_context.append("# TARGET FUNCTION TO REFACTOR:")
    if result and result.code:
        surgical_context.append(result.code)
    surgical_context.append("")

    # Add the caller for context
    surgical_context.append("# CALLER (uses calculate_tax):")
    if caller_result and caller_result.code:
        surgical_context.append(caller_result.code)

    surgical_code = '\n'.join(surgical_context)

    print(surgical_code)
    print()
    print("-" * 70)

    lines = surgical_code.strip().split('\n')
    total_chars = len(surgical_code)
    total_tokens = total_chars // 4

    print(f"Extracted lines: {len(lines)}")
    print(f"Total characters: {total_chars:,}")
    print(f"Estimated tokens: {total_tokens:,}")
    print()
    print("Benefits of this approach:")
    print("1. LLM sees ONLY the code that matters")
    print("2. Includes the caller so LLM understands usage")
    print("3. Much more tokens available for reasoning")
    print("4. Lower risk of unintended modifications")
    print()

    return {
        "lines": len(lines),
        "chars": total_chars,
        "tokens": total_tokens
    }


def demo_security_scan():
    """Show Code Scalpel's security analysis capabilities."""
    print("=" * 70)
    print("BONUS: SECURITY ANALYSIS")
    print("=" * 70)
    print()

    vulnerable_code = '''
import sqlite3
from flask import request

def get_user(username):
    conn = sqlite3.connect('users.db')
    cursor = conn.cursor()
    query = "SELECT * FROM users WHERE name = '" + username + "'"
    cursor.execute(query)
    return cursor.fetchone()

def search():
    term = request.args.get('q')
    return get_user(term)
'''

    print("Analyzing potentially vulnerable code:")
    print("-" * 70)
    print(vulnerable_code)
    print("-" * 70)

    analyzer = CodeAnalyzer()
    result = analyzer.analyze(vulnerable_code)

    vulnerabilities = result.security_issues

    if vulnerabilities:
        print()
        print(f"Found {len(vulnerabilities)} security issue(s):")
        print()
        for vuln in vulnerabilities:
            if isinstance(vuln, dict):
                print(f"  Type: {vuln.get('type', 'unknown')}")
                print(f"  Line: {vuln.get('line', 'N/A')}")
                print(f"  Message: {vuln.get('message', 'N/A')}")
                print(f"  Severity: {vuln.get('severity', 'N/A')}")
            else:
                print(f"  {vuln}")
            print()
    else:
        print("No security issues detected (basic scan)")


def main():
    print()
    print("*" * 70)
    print("*" + " " * 68 + "*")
    print("*" + "    CODE SCALPEL DEMONSTRATION: SURGICAL CODE EXTRACTION    ".center(68) + "*")
    print("*" + " " * 68 + "*")
    print("*" * 70)
    print()

    # Run demos
    without_results = demo_without_code_scalpel()
    print()
    with_results = demo_with_code_scalpel()

    # Calculate savings
    token_savings = without_results["tokens"] - with_results["tokens"]
    savings_percentage = (token_savings / without_results["tokens"]) * 100

    print("=" * 70)
    print("COMPARISON SUMMARY")
    print("=" * 70)
    print()
    print(f"{'Metric':<25} {'Without CS':<15} {'With CS':<15} {'Savings':<15}")
    print("-" * 70)
    print(f"{'Lines':<25} {without_results['lines']:<15} {with_results['lines']:<15} {without_results['lines'] - with_results['lines']:<15}")
    print(f"{'Characters':<25} {without_results['chars']:<15,} {with_results['chars']:<15,} {without_results['chars'] - with_results['chars']:<15,}")
    print(f"{'Tokens (est.)':<25} {without_results['tokens']:<15,} {with_results['tokens']:<15,} {token_savings:<15,}")
    print(f"{'Token Savings %':<25} {'-':<15} {'-':<15} {savings_percentage:.1f}%")
    print()

    # Run security demo
    demo_security_scan()

    print("=" * 70)
    print("DEMONSTRATION COMPLETE")
    print("=" * 70)
    print()
    print("Key Takeaways:")
    print(f"1. Token reduction: {savings_percentage:.1f}% fewer tokens needed")
    print(f"2. Precision: Only relevant code extracted with dependencies")
    print("3. Security: Built-in vulnerability detection")
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
