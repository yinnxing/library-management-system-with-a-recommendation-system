package com.library.booksystem.controller;


import com.library.booksystem.dto.request.UserRequest;
import com.library.booksystem.dto.response.ApiResponse;
import com.library.booksystem.dto.response.UserResponse;
import com.library.booksystem.service.UserService;
import jakarta.validation.Valid;
import lombok.AccessLevel;
import lombok.RequiredArgsConstructor;
import lombok.experimental.FieldDefaults;
import lombok.extern.slf4j.Slf4j;
import org.springframework.security.core.context.SecurityContextHolder;
import org.springframework.web.bind.annotation.*;

import java.util.List;

@RestController
@RequestMapping("/users")
@RequiredArgsConstructor
@Slf4j
@FieldDefaults(level = AccessLevel.PRIVATE, makeFinal = true)
public class UserController {
    UserService userService;

    @PostMapping
    ApiResponse<UserResponse> createUser(@RequestBody @Valid UserRequest request){
        return ApiResponse.<UserResponse>builder()
                .result(userService.createUser(request))
                .build();
    }
    @GetMapping
    ApiResponse<List<UserResponse>> getUsers(){
        var authentication = SecurityContextHolder.getContext().getAuthentication();
        authentication.getAuthorities().forEach(grantedAuthority -> log.info(grantedAuthority.getAuthority()));
        return ApiResponse.<List<UserResponse>>builder()
                .result(userService.getUsers())
                .build();
    }


    @GetMapping("/myInfo")
    ApiResponse<UserResponse> getMyInfo(){
        return ApiResponse.<UserResponse>builder()
                .result(userService.getMyInfo())
                .build();
    }



}
