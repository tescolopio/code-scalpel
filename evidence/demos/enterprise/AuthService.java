/*
 * Enterprise Demo: Service Layer
 * 
 * This file is called BY AuthController.java.
 * Code Scalpel's call graph should show this dependency.
 */
package com.demo;

public class AuthService {
    
    /**
     * Validate user credentials.
     * Called from: AuthController.login()
     */
    public boolean validate(String username, String password) {
        // Simplified validation for demo
        if (username == null || password == null) {
            return false;
        }
        
        // In real code, this would check a database
        return password.length() >= 8;
    }
    
    /**
     * Check if user has admin privileges.
     * Not called in this demo - should NOT appear in call graph.
     */
    public boolean isAdmin(String username) {
        return "admin".equals(username);
    }
}
