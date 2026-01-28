"""
Explanation Generator - Convert verification results to human-readable explanations.

This module provides formatting capabilities to convert VerificationResult objects
into human-readable text, LaTeX, and Markdown formats for display.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional

from .verification_api import VerificationResult, VerificationDomain


# =============================================================================
# Output Format Enum
# =============================================================================

class OutputFormat(str, Enum):
    """Available output formats for explanations."""
    PLAIN = "plain"
    MARKDOWN = "markdown"
    LATEX = "latex"
    HTML = "html"
    RICH = "rich"  # For rich terminal output


# =============================================================================
# Explanation Templates
# =============================================================================

TEMPLATES = {
    VerificationDomain.UNITARITY.value: {
        "verified": {
            "plain": "VERIFIED: {subject} is unitary.\n\nProof: The gate satisfies U†U = I (identity matrix).\n{proof}",
            "markdown": "## ✓ Verified: Unitary Gate\n\n**{subject}** is a valid unitary gate.\n\n### Proof\n\nThe gate satisfies the unitarity condition:\n\n$$U^\\dagger U = I$$\n\n{proof}",
            "latex": "\\textbf{{Verified:}} {subject} is unitary.\n\n\\textbf{{Proof:}} $U^\\dagger U = I$\n\n{proof}",
        },
        "failed": {
            "plain": "NOT VERIFIED: {subject} is NOT unitary.\n\nThe gate does not satisfy U†U = I.\n{proof}",
            "markdown": "## ✗ Not Verified: Not Unitary\n\n**{subject}** is NOT a unitary gate.\n\n### Details\n\nThe gate fails the unitarity condition:\n\n{proof}",
            "latex": "\\textbf{{Not Verified:}} {subject} is not unitary.\n\n{proof}",
        }
    },
    VerificationDomain.NORMALIZATION.value: {
        "verified": {
            "plain": "VERIFIED: The state is normalized.\n\nProof: The inner product ⟨ψ|ψ⟩ = 1.\n{proof}",
            "markdown": "## ✓ Verified: Normalized State\n\nThe quantum state **{subject}** is normalized.\n\n### Proof\n\n$$\\langle\\psi|\\psi\\rangle = 1$$\n\n{proof}",
            "latex": "\\textbf{{Verified:}} The state is normalized.\n\n$\\langle\\psi|\\psi\\rangle = 1$\n\n{proof}",
        },
        "failed": {
            "plain": "NOT VERIFIED: The state is NOT normalized.\n\nThe inner product ⟨ψ|ψ⟩ ≠ 1.\n{proof}",
            "markdown": "## ✗ Not Verified: Not Normalized\n\nThe quantum state **{subject}** is NOT normalized.\n\n### Details\n\n{proof}",
            "latex": "\\textbf{{Not Verified:}} The state is not normalized.\n\n{proof}",
        }
    },
    VerificationDomain.HERMITICITY.value: {
        "verified": {
            "plain": "VERIFIED: {subject} is Hermitian (self-adjoint).\n\nProof: The operator satisfies A = A† (A - A† = 0).\n{proof}",
            "markdown": "## ✓ Verified: Hermitian Operator\n\n**{subject}** is a valid Hermitian operator (observable).\n\n### Proof\n\n$$A = A^\\dagger$$\n\n{proof}",
            "latex": "\\textbf{{Verified:}} {subject} is Hermitian.\n\n$A = A^\\dagger$\n\n{proof}",
        },
        "failed": {
            "plain": "NOT VERIFIED: {subject} is NOT Hermitian.\n\nThe operator does not satisfy A = A†.\n{proof}",
            "markdown": "## ✗ Not Verified: Not Hermitian\n\n**{subject}** is NOT a Hermitian operator.\n\n### Details\n\n{proof}",
            "latex": "\\textbf{{Not Verified:}} {subject} is not Hermitian.\n\n{proof}",
        }
    },
    VerificationDomain.ENTANGLEMENT.value: {
        "verified": {
            "plain": "VERIFIED: The state is maximally entangled.\n\nProof: The von Neumann entropy S(ρ_A) = ln(d), indicating maximal entanglement.\n{proof}",
            "markdown": "## ✓ Verified: Maximally Entangled\n\nThe quantum state **{subject}** is maximally entangled.\n\n### Proof\n\nThe von Neumann entropy of the reduced density matrix:\n\n$$S(\\rho_A) = \\ln(d)$$\n\n{proof}",
            "latex": "\\textbf{{Verified:}} The state is maximally entangled.\n\n$S(\\rho_A) = \\ln(d)$\n\n{proof}",
        },
        "failed": {
            "plain": "NOT VERIFIED: The state is NOT maximally entangled.\n\nThe entropy S(ρ_A) ≠ ln(d).\n{proof}",
            "markdown": "## ✗ Not Verified: Not Maximally Entangled\n\nThe quantum state **{subject}** is NOT maximally entangled.\n\n### Details\n\n{proof}",
            "latex": "\\textbf{{Not Verified:}} The state is not maximally entangled.\n\n{proof}",
        }
    },
    VerificationDomain.BELL_STATE.value: {
        "verified": {
            "plain": "VERIFIED: Valid Bell state.\n\nThe state is normalized and maximally entangled.\n{proof}",
            "markdown": "## ✓ Verified: Valid Bell State\n\nThe quantum state **{subject}** is a valid Bell state.\n\n### Properties\n\n- ✓ Normalized\n- ✓ Maximally entangled\n\n{proof}",
            "latex": "\\textbf{{Verified:}} Valid Bell state.\n\n{proof}",
        },
        "failed": {
            "plain": "NOT VERIFIED: Invalid Bell state.\n\n{proof}",
            "markdown": "## ✗ Not Verified: Invalid Bell State\n\nThe quantum state **{subject}** is NOT a valid Bell state.\n\n### Details\n\n{proof}",
            "latex": "\\textbf{{Not Verified:}} Invalid Bell state.\n\n{proof}",
        }
    },
    VerificationDomain.CHSH.value: {
        "verified": {
            "plain": "VERIFIED: CHSH inequality violation detected!\n\nThe quantum system violates the classical bound |S| ≤ 2.\n{proof}",
            "markdown": "## ✓ Verified: CHSH Violation\n\nQuantum violation of Bell's inequality detected!\n\n### Result\n\n$$|S| > 2$$\n\n{proof}",
            "latex": "\\textbf{{Verified:}} CHSH violation.\n\n$|S| > 2$\n\n{proof}",
        },
        "failed": {
            "plain": "NOT VERIFIED: No CHSH violation.\n\nThe system stays within the classical bound |S| ≤ 2.\n{proof}",
            "markdown": "## ✗ Not Verified: No CHSH Violation\n\nNo quantum violation detected.\n\n### Details\n\n$$|S| \\leq 2$$\n\n{proof}",
            "latex": "\\textbf{{Not Verified:}} No CHSH violation.\n\n{proof}",
        }
    },
    "general": {
        "verified": {
            "plain": "VERIFIED\n\n{explanation}\n{proof}",
            "markdown": "## ✓ Verified\n\n{explanation}\n\n{proof}",
            "latex": "\\textbf{{Verified}}\n\n{explanation}\n\n{proof}",
        },
        "failed": {
            "plain": "NOT VERIFIED\n\n{explanation}\n{proof}",
            "markdown": "## ✗ Not Verified\n\n{explanation}\n\n{proof}",
            "latex": "\\textbf{{Not Verified}}\n\n{explanation}\n\n{proof}",
        }
    }
}


# =============================================================================
# Main Generator Class
# =============================================================================

class ExplanationGenerator:
    """
    Generator for human-readable explanations of verification results.

    Converts VerificationResult objects into formatted text suitable
    for display in CLI, web interfaces, or documentation.

    Example:
        >>> generator = ExplanationGenerator()
        >>> result = api.verify_gate("H")
        >>> explanation = generator.generate(result)
        >>> print(explanation)
    """

    def __init__(self, default_format: OutputFormat = OutputFormat.PLAIN):
        """
        Initialize the generator.

        Args:
            default_format: Default output format to use
        """
        self.default_format = default_format
        self._templates = TEMPLATES

    def generate(
        self,
        result: VerificationResult,
        format: Optional[OutputFormat] = None,
        subject: str = ""
    ) -> str:
        """
        Generate a human-readable explanation from a verification result.

        Args:
            result: The VerificationResult to explain
            format: Output format (defaults to self.default_format)
            subject: Optional subject name for the template

        Returns:
            Formatted explanation string
        """
        fmt = format or self.default_format
        domain = result.domain
        status = "verified" if result.verified else "failed"

        # Get the appropriate template
        templates = self._templates.get(domain, self._templates["general"])
        template = templates.get(status, {}).get(fmt.value, templates[status]["plain"])

        # Extract subject from details if not provided
        if not subject:
            subject = result.details.get("input_gate", "")
            subject = subject or result.details.get("input_state", "")
            subject = subject or result.details.get("input_operator", "")
            subject = subject or "the input"

        # Format the template
        return template.format(
            subject=subject,
            proof=result.symbolic_proof,
            explanation=result.explanation,
            **result.details
        )

    def to_plain(self, result: VerificationResult, subject: str = "") -> str:
        """Generate plain text explanation."""
        return self.generate(result, OutputFormat.PLAIN, subject)

    def to_markdown(self, result: VerificationResult, subject: str = "") -> str:
        """Generate Markdown explanation."""
        return self.generate(result, OutputFormat.MARKDOWN, subject)

    def to_latex(self, result: VerificationResult, subject: str = "") -> str:
        """Generate LaTeX explanation."""
        return self.generate(result, OutputFormat.LATEX, subject)

    def to_html(self, result: VerificationResult, subject: str = "") -> str:
        """
        Generate HTML explanation.

        Converts Markdown to basic HTML.
        """
        md = self.to_markdown(result, subject)

        # Basic Markdown to HTML conversion
        html = md
        html = html.replace("## ✓", "<h2 class='verified'>✓")
        html = html.replace("## ✗", "<h2 class='failed'>✗")
        html = html.replace("##", "<h2>")
        html = html.replace("\n\n", "</h2>\n<p>", 1)
        html = html.replace("\n\n", "</p>\n<p>")
        html = html.replace("**", "<strong>").replace("**", "</strong>")
        html = html.replace("$$", "<div class='math'>").replace("$$", "</div>")
        html = html.replace("- ✓", "<li class='pass'>✓")
        html = html.replace("- ✗", "<li class='fail'>✗")

        return f"<div class='verification-result'>\n{html}\n</div>"

    def to_rich(self, result: VerificationResult, subject: str = "") -> str:
        """
        Generate Rich-formatted explanation for terminal output.

        Uses Rich markup for colored terminal output.
        """
        if result.verified:
            status = "[bold green]✓ VERIFIED[/bold green]"
        else:
            status = "[bold red]✗ NOT VERIFIED[/bold red]"

        domain_display = result.domain.replace("_", " ").title()

        output = f"{status}\n\n"
        output += f"[bold]Domain:[/bold] {domain_display}\n"
        output += f"[bold]Confidence:[/bold] {result.confidence * 100:.0f}%\n\n"
        output += f"[italic]{result.explanation}[/italic]\n\n"

        if result.symbolic_proof:
            output += f"[bold]Symbolic Proof:[/bold]\n{result.symbolic_proof}\n"

        return output

    def format_proof_step(
        self,
        step_num: int,
        description: str,
        expression: str,
        format: OutputFormat = OutputFormat.PLAIN
    ) -> str:
        """
        Format a single proof step.

        Useful for building step-by-step proof explanations.
        """
        if format == OutputFormat.MARKDOWN:
            return f"{step_num}. **{description}**\n   $${{ {expression} }}$$\n"
        elif format == OutputFormat.LATEX:
            return f"\\textbf{{{step_num}. {description}}}\n$${expression}$$\n"
        else:
            return f"{step_num}. {description}\n   {expression}\n"

    def generate_summary(self, results: list[VerificationResult]) -> str:
        """
        Generate a summary of multiple verification results.

        Args:
            results: List of VerificationResult objects

        Returns:
            Summary string with statistics
        """
        total = len(results)
        verified = sum(1 for r in results if r.verified)
        failed = total - verified

        summary = f"Verification Summary\n"
        summary += f"{'=' * 40}\n"
        summary += f"Total claims: {total}\n"
        summary += f"Verified:     {verified} ({verified/total*100:.1f}%)\n"
        summary += f"Failed:       {failed} ({failed/total*100:.1f}%)\n"

        if failed > 0:
            summary += f"\nFailed verifications:\n"
            for i, r in enumerate(results):
                if not r.verified:
                    summary += f"  - {r.domain}: {r.explanation[:50]}...\n"

        return summary


# =============================================================================
# Module-level convenience functions
# =============================================================================

def explain(result: VerificationResult, format: OutputFormat = OutputFormat.PLAIN) -> str:
    """Generate an explanation using the default generator."""
    return ExplanationGenerator().generate(result, format)


def explain_markdown(result: VerificationResult) -> str:
    """Generate a Markdown explanation."""
    return ExplanationGenerator().to_markdown(result)


def explain_latex(result: VerificationResult) -> str:
    """Generate a LaTeX explanation."""
    return ExplanationGenerator().to_latex(result)
