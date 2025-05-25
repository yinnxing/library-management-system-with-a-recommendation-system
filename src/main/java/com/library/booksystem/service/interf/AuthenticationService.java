package com.library.booksystem.service.interf;

import com.library.booksystem.dto.request.AuthenticationRequest;
import com.library.booksystem.dto.request.IntrospectRequest;
import com.library.booksystem.dto.request.LogoutRequest;
import com.library.booksystem.dto.request.RefreshRequest;
import com.library.booksystem.dto.response.AuthenticationResponse;
import com.library.booksystem.dto.response.IntrospectResponse;
import com.library.booksystem.dto.response.TokenResponse;
import com.library.booksystem.model.User;
import com.nimbusds.jose.JOSEException;
import com.nimbusds.jwt.JWTClaimsSet;
import com.nimbusds.jwt.SignedJWT;


import java.text.ParseException;

public interface AuthenticationService {
    public TokenResponse refreshToken(String request) throws ParseException, JOSEException;
    public IntrospectResponse introspect(IntrospectRequest request) throws ParseException, JOSEException;
    public void Logout(LogoutRequest request) throws ParseException, JOSEException;
    public AuthenticationResponse authenticate(AuthenticationRequest request);

    public String generateAccessToken(User user);
    public String generateRefreshToken(User user);

    public SignedJWT verifyToken(String token, boolean isRefresh) throws JOSEException, ParseException;




    }
