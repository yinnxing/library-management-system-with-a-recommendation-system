package com.library.booksystem.exception;

import lombok.Getter;
import org.springframework.http.HttpStatus;
import org.springframework.http.HttpStatusCode;

@Getter
public enum ErrorCode {
    INVALID_KEY(0000, "Uncategorized error", HttpStatus.BAD_REQUEST),
    UNCATEGORIZED_EXCEPTION(9999, "Uncategorized exception", HttpStatus.INTERNAL_SERVER_ERROR),
    USER_EXISTED(1001, "User existed",HttpStatus.BAD_REQUEST ),
    USERNAME_INVALID(1002, "Username must be at least {min} characters", HttpStatus.BAD_REQUEST),
    PASSWORD_INVALID(1003, "Password must be at least {min} characters", HttpStatus.BAD_REQUEST),
    DOB_INVALID(1004, "Your age must be at least {min}", HttpStatus.BAD_REQUEST),
    USER_NOT_EXIST(1005, "User not exist", HttpStatus.NOT_FOUND),
    ROLE_NOT_FOUND(1006, "Role not found", HttpStatus.NOT_FOUND),
    UNAUTHENTICATED(1007, "Unauthenticated", HttpStatus.UNAUTHORIZED),
    UNAUTHORIZED(1008, "You do not have permission", HttpStatus.FORBIDDEN),
    REFRESH_TOKEN_EXPIRED(1012, "", HttpStatus.UNAUTHORIZED),
    BOOK_EXISTED(1009, "Book existed", HttpStatus.BAD_REQUEST),
    BOOK_NOT_FOUND(1010, "Book not found", HttpStatus.NOT_FOUND),
    BOOK_NOT_AVAILABLE(1011, "book is not available", HttpStatus.BAD_REQUEST),
    PAGE_NUMBER_INVALID(1012, "page number must be greater than 0", HttpStatus.BAD_REQUEST),
    SIZE_INVALID(1013, "size must be greater than 0", HttpStatus.BAD_REQUEST),
    OAUTH2_PROVIDER_NOT_SUPPORTED(1014, "OAuth2 provider not supported", HttpStatus.BAD_REQUEST),
    REVIEW_NOT_FOUND(404, "Review not found", HttpStatus.NOT_FOUND)


    ;
    private int code;
    private String message;
    private HttpStatusCode statusCode;


    ErrorCode(int code, String message, HttpStatusCode statusCode) {
        this.code = code;
        this.message = message;
        this.statusCode = statusCode;
    }



}
