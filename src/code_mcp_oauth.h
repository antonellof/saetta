// Saetta Code's OAuth sign-in for Streamable HTTP MCP servers (code_mcp_oauth.c): authorization
// server discovery (RFC 9728 protected resource metadata, RFC 8414 and OpenID discovery), dynamic
// client registration (RFC 7591) or a Client ID Metadata Document, the authorization code flow with
// PKCE through the system browser and a loopback redirect listener, refresh, and pi's credential
// store (<agent dir>/mcp-auth.json, same keys and format, so pi and Saetta Code share sign-ins).
// Adapted from pi (MIT, Copyright (c) 2025 Mario Zechner; github.com/earendil-works/pi at v1.5.2,
// b4afddd): packages/mcp/src/oauth/{auth-provider,callback,client-settings,discovery,errors,flow,
// loopback,provider,sign-in,types}.ts (themselves adapted from modelcontextprotocol/typescript-sdk
// v1.29.0, MIT, Copyright (c) 2024 Anthropic, PBC) and packages/coding-agent/src/extensions/mcp/
// oauth.ts, packages/ai/src/utils/oauth-page.ts.
//
// Every function blocks (the HTTP requests to the authorization server, 15 s each at most) and is
// thread safe; none is called on the UI thread.
#ifndef SAETTA_CODE_MCP_OAUTH_H
#define SAETTA_CODE_MCP_OAUTH_H

#include "base.h"

#include <pthread.h>

// McpOAuthSettings: a server's "oauth" object (strings NULL when absent; clientSecret resolved).
typedef struct {
    const char *client_id, *client_secret, *callback_url, *scope, *client_name, *client_registration;
    const char *auth_server_metadata_url;
    int callback_port; // 0: none
} McpOAuthSettings;

// The settings, resolved when needed (oauthSettings): false with *err (malloc'd) when the client
// secret does not resolve. Strings in the arena.
typedef bool (*McpOAuthSettingsFn)(void *ctx, Arena *a, McpOAuthSettings *out, char **err);

// createMcpAuthProvider for one server (name, url as configured), its state in
// <agent_dir>/mcp-auth.json. ctx stays valid as long as the provider.
typedef struct McpAuth McpAuth;
McpAuth *mcp_auth_new(const char *agent_dir, const char *name, const char *url, McpOAuthSettingsFn settings, void *ctx);

// token(): the stored access token (malloc'd, NULL when none), refreshed first when it expires in
// less than 30 s and a refresh token is stored (a failed refresh sends the old one).
char *mcp_auth_token(McpAuth *m);
// onUnauthorized after a 401, or a 403 whose challenge is insufficient_scope: the challenge kept
// for the sign-in, then a refresh of stale_token (the rejected one, NULL when none was sent).
// MCP_AUTH_RETRY: try the request again; MCP_AUTH_REQUIRED: the user has to sign in
// (McpOAuthAuthorizationRequiredError); MCP_AUTH_FAILED: the refresh failed, *err says why.
enum { MCP_AUTH_RETRY = 0, MCP_AUTH_REQUIRED = 1, MCP_AUTH_FAILED = 2 };
int mcp_auth_unauthorized(McpAuth *m, const char *www_authenticate, const char *stale_token, char **err);
// settled(): waits until no refresh runs (shutdown keeps rotated tokens).
void mcp_auth_settled(McpAuth *m);
// connection.challenge = undefined (a sign-in answered it)
void mcp_auth_clear_challenge(McpAuth *m);

// The stored tokens as JSON text ("null" when none), for noticing sign-ins of other processes
// (index.ts storedTokens). malloc'd.
char *mcp_auth_stored_tokens(McpAuth *m);
// McpOAuthCredentialStore.remove: whether credentials were stored.
bool mcp_auth_remove(McpAuth *m);

// signInMcpServer: the stored refresh token when it still works, else the browser authorization
// code flow against a loopback callback or a pasted redirect URL. The sign-in's side of the UI:
typedef struct {
    volatile int cancel; // set to stop the sign-in at any step (McpSignInCancelledError)
    // showAuthorizationUrl: the URL to show and open in the browser (on the sign-in's thread)
    void (*show_url)(void *ctx, const char *url);
    // the authorization response arrived (from the browser or pasted): "Connecting…"
    void (*responded)(void *ctx);
    void *ctx;
    // the redirect URL the user pasted (mcp_sign_in_paste)
    pthread_mutex_t mu;
    char *pasted;
    bool gave_up;
} McpSignIn;
void mcp_sign_in_init(McpSignIn *si);
void mcp_sign_in_destroy(McpSignIn *si);
// promptForRedirectUrl's answer: the pasted URL, or NULL / empty when the user cancelled.
void mcp_sign_in_paste(McpSignIn *si, const char *url);
// MCP_SIGNIN_OK, MCP_SIGNIN_CANCELLED, or MCP_SIGNIN_FAILED with *err (malloc'd, the error's
// message without pi's "Sign-in failed: " prefix).
enum { MCP_SIGNIN_OK = 0, MCP_SIGNIN_CANCELLED = 1, MCP_SIGNIN_FAILED = 2 };
int mcp_auth_sign_in(McpAuth *m, McpSignIn *si, char **err);

// The test hook for the browser step: with SAETTA_MCP_OAUTH_FETCH set, the authorization URL is
// fetched here (redirects followed, so an authorization server that approves at once reaches the
// loopback callback) instead of being opened with `open`.
void mcp_oauth_open_url(const char *url);

#endif
