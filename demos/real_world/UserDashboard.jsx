/**
 * React Component Analysis Demo
 * 
 * This demo shows Code Scalpel analyzing JavaScript/React code
 * for security issues and code quality.
 * 
 * Target: Frontend developers
 * Proves: Polyglot analysis works on real React patterns
 * 
 * Run:
 *     code-scalpel analyze demos/real_world/UserDashboard.jsx
 */

import React, { useState, useEffect } from 'react';

// =============================================================================
// VULNERABLE: XSS via dangerouslySetInnerHTML
// =============================================================================
function UserProfile({ user }) {
    /**
     * Renders user bio with potential XSS.
     * 
     * VULNERABILITY: dangerouslySetInnerHTML with user-controlled content.
     * Code Scalpel should detect: user.bio -> dangerouslySetInnerHTML
     */
    return (
        <div className="profile">
            <h1>{user.name}</h1>
            {/* BAD: XSS vulnerability */}
            <div 
                className="bio"
                dangerouslySetInnerHTML={{ __html: user.bio }}  // CWE-79
            />
        </div>
    );
}

// =============================================================================
// SAFE: Proper text rendering
// =============================================================================
function UserProfileSafe({ user }) {
    /**
     * Safe user profile rendering.
     * React automatically escapes text content.
     */
    return (
        <div className="profile">
            <h1>{user.name}</h1>
            {/* GOOD: React escapes this automatically */}
            <div className="bio">{user.bio}</div>
        </div>
    );
}

// =============================================================================
// VULNERABLE: eval() with user input
// =============================================================================
function Calculator({ expression }) {
    /**
     * Calculator that evaluates user expressions.
     * 
     * VULNERABILITY: eval() on user input.
     * Code Scalpel should detect: expression -> eval()
     */
    const [result, setResult] = useState(null);
    
    const calculate = () => {
        try {
            // BAD: Code injection via eval
            const value = eval(expression);  // CWE-94: Code Injection
            setResult(value);
        } catch (e) {
            setResult('Error');
        }
    };
    
    return (
        <div>
            <button onClick={calculate}>Calculate</button>
            <span>Result: {result}</span>
        </div>
    );
}

// =============================================================================
// SAFE: Proper math parsing
// =============================================================================
function CalculatorSafe({ expression }) {
    /**
     * Safe calculator using a math library.
     */
    const [result, setResult] = useState(null);
    
    const calculate = () => {
        try {
            // GOOD: Use a safe math parser (e.g., mathjs)
            // const value = math.evaluate(expression);
            
            // Or validate strictly
            if (/^[\d+\-*/().\s]+$/.test(expression)) {
                const value = Function(`"use strict"; return (${expression})`)();
                setResult(value);
            } else {
                setResult('Invalid expression');
            }
        } catch (e) {
            setResult('Error');
        }
    };
    
    return (
        <div>
            <button onClick={calculate}>Calculate</button>
            <span>Result: {result}</span>
        </div>
    );
}

// =============================================================================
// VULNERABLE: URL manipulation
// =============================================================================
function ExternalLink({ url, children }) {
    /**
     * Link component with open redirect vulnerability.
     * 
     * VULNERABILITY: Unvalidated URL from props.
     */
    // BAD: Open redirect - url could be javascript: or data:
    return (
        <a href={url} target="_blank" rel="noopener">
            {children}
        </a>
    );
}

// =============================================================================
// SAFE: URL validation
// =============================================================================
function ExternalLinkSafe({ url, children }) {
    /**
     * Safe link component with URL validation.
     */
    const safeUrl = (() => {
        try {
            const parsed = new URL(url);
            // GOOD: Only allow http/https
            if (parsed.protocol === 'http:' || parsed.protocol === 'https:') {
                return url;
            }
            return '#';
        } catch {
            return '#';
        }
    })();
    
    return (
        <a href={safeUrl} target="_blank" rel="noopener noreferrer">
            {children}
        </a>
    );
}

// =============================================================================
// Main Dashboard Component
// =============================================================================
function UserDashboard() {
    const [user, setUser] = useState(null);
    const [loading, setLoading] = useState(true);
    
    useEffect(() => {
        // Fetch user data
        fetch('/api/user/me')
            .then(res => res.json())
            .then(data => {
                setUser(data);
                setLoading(false);
            });
    }, []);
    
    if (loading) return <div>Loading...</div>;
    if (!user) return <div>Not logged in</div>;
    
    return (
        <div className="dashboard">
            <UserProfile user={user} />
            <Calculator expression={user.lastExpression} />
            <ExternalLink url={user.website}>
                Visit Website
            </ExternalLink>
        </div>
    );
}

export default UserDashboard;

// =============================================================================
// Expected Code Scalpel Output:
//
// Found 3 vulnerability(ies):
//   1. XSS (CWE-79) at line 30
//      dangerouslySetInnerHTML with user-controlled content
//   2. Code Injection (CWE-94) at line 62
//      eval() called with user input
//   3. Open Redirect at line 108
//      Unvalidated URL in href attribute
//
// Code Quality:
//   - Functions: 8
//   - Components: 6
//   - Hooks used: useState, useEffect
//   - Complexity: Low-Medium
// =============================================================================
