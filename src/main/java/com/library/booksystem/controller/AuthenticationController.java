package com.library.booksystem.controller;


import com.library.booksystem.dto.request.AuthenticationRequest;
import com.library.booksystem.dto.request.IntrospectRequest;
import com.library.booksystem.dto.request.LogoutRequest;
import com.library.booksystem.dto.response.ApiResponse;
import com.library.booksystem.dto.response.AuthenticationResponse;
import com.library.booksystem.dto.response.IntrospectResponse;
import com.library.booksystem.dto.response.TokenResponse;
import com.library.booksystem.exception.AppException;
import com.library.booksystem.exception.ErrorCode;
import com.library.booksystem.service.interf.AuthenticationService;
import com.nimbusds.jose.JOSEException;
import jakarta.servlet.http.Cookie;
import jakarta.servlet.http.HttpServletRequest;
import lombok.AccessLevel;
import lombok.RequiredArgsConstructor;
import lombok.experimental.FieldDefaults;
import org.springframework.web.bind.annotation.*;

import java.text.ParseException;

@RestController
@RequestMapping("/auth")
@RequiredArgsConstructor
@FieldDefaults(level = AccessLevel.PRIVATE, makeFinal = true)
public class AuthenticationController {
    AuthenticationService authenticationService;
    @PostMapping("/token")
    ApiResponse<AuthenticationResponse> authenticate(@RequestBody AuthenticationRequest request){
        var result = authenticationService.authenticate(request);
        return ApiResponse.<AuthenticationResponse>builder()
                .result(result)
                .build();
    }
    @PostMapping("/refresh")
    ApiResponse<TokenResponse> refresh(HttpServletRequest request) throws ParseException, JOSEException {
        Cookie[] cookies = request.getCookies();
        String refreshToken = null;

        if (cookies != null) {
            for (Cookie cookie : cookies) {
                if ("refreshToken".equals(cookie.getName())) {
                    refreshToken = cookie.getValue();
                    break;
                }
            }
        }

        if (refreshToken == null) {
           throw new AppException(ErrorCode.UNAUTHENTICATED);
        }


        TokenResponse tokenResponse = authenticationService.refreshToken(refreshToken);
        return ApiResponse.<TokenResponse>builder()
                .result(tokenResponse)
                .build();
    }


    @PostMapping("/introspect")
    ApiResponse<IntrospectResponse> authenticate(@RequestBody IntrospectRequest request) throws ParseException, JOSEException {
        var result = authenticationService.introspect(request);
        return ApiResponse.<IntrospectResponse>builder()
                .result(result)
                .build();
    }
    @PostMapping("/logout")
    ApiResponse<Void> logout(@RequestBody LogoutRequest request) throws ParseException, JOSEException {
        authenticationService.Logout(request);
        return ApiResponse.<Void>builder().build();
    }



}
