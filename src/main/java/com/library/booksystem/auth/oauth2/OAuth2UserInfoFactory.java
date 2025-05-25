package com.library.booksystem.auth.oauth2;

import com.library.booksystem.exception.AppException;
import com.library.booksystem.exception.ErrorCode;

import java.util.Map;

public class OAuth2UserInfoFactory {
    public static OAuth2UserInfo getOAuth2UserInfo(String registrationId, Map<String, Object> attributes) {
        if(registrationId.equalsIgnoreCase("google")) {
            return new GoogleOAuth2UserInfo(attributes);
        } else if (registrationId.equalsIgnoreCase("facebook")) {
            return new FacebookOAuth2UserInfo(attributes);
        } else {
            throw new AppException(ErrorCode.OAUTH2_PROVIDER_NOT_SUPPORTED);
        }
    }
} 