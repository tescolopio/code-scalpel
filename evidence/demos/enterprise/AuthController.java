/*
 * Enterprise Demo: Cross-File Call Graph
 * 
 * This demo proves Code Scalpel can analyze Enterprise Java projects
 * and build call graphs across multiple files.
 * 
 * Run:
 *     code-scalpel analyze demos/enterprise/AuthController.java
 * 
 * Expected: Call graph shows AuthController.login() -> AuthService.validate()
 */
package com.demo;

public class AuthController {
    private AuthService service = new AuthService();
    
    /**
     * Handle login request.
     * Code Scalpel should detect the cross-file call to AuthService.validate()
     */
    public void login(String user, String pass) {
        // Cross-file dependency - calls method in AuthService.java
        if (service.validate(user, pass)) {
            System.out.println("Welcome, " + user);
            auditLog("LOGIN_SUCCESS", user);
        } else {
            System.out.println("Access Denied");
            auditLog("LOGIN_FAILURE", user);
        }
    }
    
    /**
     * Internal audit logging.
     * Code Scalpel should detect this as an internal call.
     */
    private void auditLog(String event, String user) {
        System.out.println("[AUDIT] " + event + " for " + user);
    }
    
    /**
     * Logout handler - no cross-file calls.
     */
    public void logout(String user) {
        System.out.println("Goodbye, " + user);
        auditLog("LOGOUT", user);
    }
}
