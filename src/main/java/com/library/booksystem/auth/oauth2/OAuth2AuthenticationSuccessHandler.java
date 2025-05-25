package com.library.booksystem.auth.oauth2;

import com.library.booksystem.auth.JwtUtils;
import com.library.booksystem.model.User;
import com.library.booksystem.repository.UserRepository;
import com.library.booksystem.service.interf.AuthenticationService;
import jakarta.servlet.ServletException;
import jakarta.servlet.http.Cookie;
import jakarta.servlet.http.HttpServletRequest;
import jakarta.servlet.http.HttpServletResponse;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.security.core.Authentication;
import org.springframework.security.web.authentication.SimpleUrlAuthenticationSuccessHandler;
import org.springframework.stereotype.Component;
import org.springframework.web.util.UriComponentsBuilder;

import java.io.IOException;
import java.util.Optional;

@Component
@RequiredArgsConstructor
@Slf4j
public class OAuth2AuthenticationSuccessHandler extends SimpleUrlAuthenticationSuccessHandler {

    private final AuthenticationService authenticationService;
    private final UserRepository userRepository;
    
    @Value("${app.oauth2.redirectUri}")
    private String redirectUri;

    @Override
    public void onAuthenticationSuccess(HttpServletRequest request, HttpServletResponse response, Authentication authentication) 
            throws IOException, ServletException {
        
        CustomOAuth2User oAuth2User = (CustomOAuth2User) authentication.getPrincipal();
        
        Optional<User> userOptional = userRepository.findByEmail(oAuth2User.getEmail());
        if (!userOptional.isPresent()) {
            log.error("User not found with email: {}", oAuth2User.getEmail());
            response.sendRedirect("/login?error=user_not_found");
            return;
        }
        
        User user = userOptional.get();
        
        // Generate JWT tokens
        String accessToken = authenticationService.generateAccessToken(user);
        String refreshToken = authenticationService.generateRefreshToken(user);
        
        // Set refresh token as HTTP-only cookie
        Cookie refreshTokenCookie = new Cookie("refreshToken", refreshToken);
        refreshTokenCookie.setHttpOnly(true);
        refreshTokenCookie.setPath("/");
        refreshTokenCookie.setMaxAge(604800); // 7 days
        response.addCookie(refreshTokenCookie);
        
        // Redirect to frontend with access token
        String targetUrl = UriComponentsBuilder.fromUriString(redirectUri)
                .queryParam("token", accessToken)
                .build().toUriString();
        
        getRedirectStrategy().sendRedirect(request, response, targetUrl);
    }
} 