package com.library.booksystem.service.impl;

import com.library.booksystem.dto.request.AuthenticationRequest;
import com.library.booksystem.dto.request.IntrospectRequest;
import com.library.booksystem.dto.request.LogoutRequest;
import com.library.booksystem.dto.response.AuthenticationResponse;
import com.library.booksystem.dto.response.IntrospectResponse;
import com.library.booksystem.dto.response.TokenResponse;
import com.library.booksystem.exception.AppException;
import com.library.booksystem.exception.ErrorCode;
import com.library.booksystem.model.InvalidedToken;
import com.library.booksystem.model.User;
import com.library.booksystem.repository.InvalidedTokenRepository;
import com.library.booksystem.repository.UserRepository;
import com.library.booksystem.service.interf.AuthenticationService;
import com.nimbusds.jose.*;
import com.nimbusds.jose.crypto.MACSigner;
import com.nimbusds.jose.crypto.MACVerifier;
import com.nimbusds.jwt.JWTClaimsSet;
import com.nimbusds.jwt.SignedJWT;
import lombok.AccessLevel;
import lombok.experimental.FieldDefaults;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.security.crypto.bcrypt.BCryptPasswordEncoder;
import org.springframework.security.crypto.password.PasswordEncoder;
import org.springframework.stereotype.Service;


import java.text.ParseException;
import java.util.Date;
import java.util.UUID;

@Service
@Slf4j
@FieldDefaults(level = AccessLevel.PRIVATE)
public class AuthenticationServiceImp implements AuthenticationService {
    @Autowired
    UserRepository userRepository;
    @Autowired
    InvalidedTokenRepository invalidedTokenRepository;
    @Value("${jwt.valid.duration}")
    Long VALID_DURATION;
    @Value("${jwt.refreshable.duration}")
    Long REFRESHABLE_DURATION;
    @Value("${jwt.signer.key}")
    String SIGNER_KEY;

    public TokenResponse refreshToken(String request) throws ParseException, JOSEException {
        var signedToken = verifyToken(request, true);

        var jit = signedToken.getJWTClaimsSet().getJWTID();
        var expiryTime = signedToken.getJWTClaimsSet().getExpirationTime();
        if (expiryTime != null && expiryTime.before(new Date())) {
            InvalidedToken invalidedToken = InvalidedToken.builder()
                    .id(jit)
                    .expiryTime(expiryTime)
                    .build();
            invalidedTokenRepository.save(invalidedToken);
            throw new AppException(ErrorCode.REFRESH_TOKEN_EXPIRED);
        }

        String email = signedToken.getJWTClaimsSet().getSubject();
        User user = userRepository.findByEmail(email).orElseThrow(
                () -> new AppException(ErrorCode.UNAUTHENTICATED)
        );
        var token = generateAccessToken(user);
        return TokenResponse.builder()
                .accessToken(token)
                .build();
    }

    public IntrospectResponse introspect(IntrospectRequest request) throws JOSEException, ParseException {
        var token = request.getToken();
        boolean isValid = true;
        try {
            verifyToken(token, false);
        } catch (AppException e) {
            isValid = false;
        }
        return IntrospectResponse.builder()
                .valid(isValid)
                .build();
    }

    public void Logout(LogoutRequest request) throws ParseException, JOSEException {
        try {
            var signToken = verifyToken(request.getToken(), true);
            String jit = signToken.getJWTClaimsSet().getJWTID();
            InvalidedToken invalidedToken = InvalidedToken.builder()
                    .id(jit)
                    .expiryTime(signToken.getJWTClaimsSet().getExpirationTime())
                    .build();
            invalidedTokenRepository.save(invalidedToken);
        } catch (AppException e) {
            log.info("Token has already expired");
        }
    }

    public AuthenticationResponse authenticate(AuthenticationRequest request) {
        PasswordEncoder passwordEncoder = new BCryptPasswordEncoder(10);
        var user = userRepository.findByEmail(request.getEmail())
                .orElseThrow(() -> new AppException(ErrorCode.USER_NOT_EXIST));
        boolean authenticated = passwordEncoder.matches(request.getPassword(), user.getPassword());

        if (!authenticated)
            throw new AppException(ErrorCode.UNAUTHENTICATED);
        var accessToken = generateAccessToken(user);
        var refreshToken = generateRefreshToken(user);

        return AuthenticationResponse.builder()
                .accessToken(accessToken)
                .refreshToken(refreshToken)
                .authenticated(true)
                .build();
    }

    public String generateAccessToken(User user) { JWTClaimsSet jwtClaimsSet = new JWTClaimsSet.Builder()
            .subject(user.getEmail())
            .issuer("lib.com")
            .issueTime(new Date())
            .expirationTime(new Date(new Date().getTime() + VALID_DURATION))
            .jwtID(UUID.randomUUID().toString())
            .claim("userId", user.getUserId())
            .claim("role", user.getRole())
            .build();
        JWSObject jwsObject = new JWSObject(
                new JWSHeader(JWSAlgorithm.HS512),
                new Payload(jwtClaimsSet.toJSONObject())
        );

        try {
            jwsObject.sign(new MACSigner(SIGNER_KEY.getBytes()));
            return jwsObject.serialize();
        } catch (JOSEException e) {
            log.error("Cannot create token" + e);
            throw new RuntimeException(e);
        }
    }
    public String generateRefreshToken(User user) { JWTClaimsSet jwtClaimsSet = new JWTClaimsSet.Builder()
            .subject(user.getEmail())
            .issuer("lib.com")
            .issueTime(new Date())
            .expirationTime(new Date(new Date().getTime() + REFRESHABLE_DURATION))
            .jwtID(UUID.randomUUID().toString())
            .claim("userId", user.getUserId())
            .claim("role", user.getRole())
            .build();
        JWSObject jwsObject = new JWSObject(
                new JWSHeader(JWSAlgorithm.HS512),
                new Payload(jwtClaimsSet.toJSONObject())
        );

        try {
            jwsObject.sign(new MACSigner(SIGNER_KEY.getBytes()));
            return jwsObject.serialize();
        } catch (JOSEException e) {
            log.error("Cannot create token" + e);
            throw new RuntimeException(e);
        }
    }

    public SignedJWT verifyToken(String token, boolean isRefresh) throws JOSEException, ParseException {
        JWSVerifier verifier = new MACVerifier(SIGNER_KEY.getBytes());
        SignedJWT signedJWT = SignedJWT.parse(token);
        Date expireTime = (isRefresh) ?
                Date.from(signedJWT.getJWTClaimsSet().getIssueTime().toInstant().plusSeconds(REFRESHABLE_DURATION))
                : signedJWT.getJWTClaimsSet().getExpirationTime();

        var verified = signedJWT.verify(verifier);
        if (!(verified && expireTime.after(new Date()))) {
            throw new AppException(ErrorCode.UNAUTHENTICATED);
        }
        if (invalidedTokenRepository
                .existsById(signedJWT.getJWTClaimsSet().getJWTID())) {
            throw new AppException(ErrorCode.UNAUTHENTICATED);
        }

        return signedJWT;
    }




}
